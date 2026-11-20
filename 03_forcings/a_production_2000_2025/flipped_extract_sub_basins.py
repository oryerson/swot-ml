"""Extract daily ERA5-Land weather for every sub-basin polygon as area-weighted means and variances.

This is the first forcing pass of the pipeline (2000-2025 in the paper, on the first graph
version). merge_era5_conus.py then writes the result into the Zarr data store.

Reads
  --basin-file   GeoParquet table of sub-basin polygons. The index is the sub-basin (node) id and
                 the column `outlet_id` names the basin.
  ERA5-Land monthly NetCDF files on the 0.1 degree grid, under <NAS_ICE>/data/ERA5-Land/
                 (NAS_ICE comes from swotgat.paths): accumulated/<year>_<month>.nc and
                 average/<year>_<month>.nc.

Writes
  --save-dir     a BasinDataLake (class of the vendored swot_ml package): Parquet files of the
                 dynamic source "era5", partitioned by basin, appended batch by batch. Each
                 ERA5-Land variable gives two columns, <variable>_mean and <variable>_var.

Usage
  python flipped_extract_sub_basins.py --basin-file FILE --save-dir DIR
         [--n-workers 20] [--batch-size 8] [--start-date 1980-01-01] [--end-date 2025-12-31]
  The script starts its own Dask cluster on Slurm (partition "cpu", one core and 32 GB per
  worker, 2-day limit) and submits one task per spatial batch. Each worker changes to
  $SWOTGAT_WORK/swot-ml and activates the virtual environment there. The `data` module must be
  importable (vendor/swot_ml/src on the Python path).

Method
  1. Within each basin, group the sub-basins into spatially compact batches of about
     --batch-size polygons (K-Means on the polygon centroids). Each batch has one bounding box,
     padded by 0.1 degree and rounded outward to 0.1 degree.
  2. For every polygon of a batch, compute once the fraction of each grid cell that the polygon
     covers. The fast method rasterizes the polygon on a grid 10 times finer than ERA5-Land and
     averages back, so fractions are multiples of 1/100. A polygon too small to contain the
     center of any fine cell gets zero coverage this way; for those the script falls back to
     the exact polygon-cell intersection area.
  3. Loop over months (outer) and polygons (inner): open the two monthly files once, cut them
     to the batch bounding box, and take the coverage-weighted mean and variance of every
     variable at every time step.

Notes
  The code moves the basin with outlet USGS-07374000 (the Mississippi) to the top of the
  sub-basin table. The batches are then built with groupby("outlet_id"), which sorts basins by
  id, so tasks are submitted in sorted basin order.
  The script does not skip sub-basins that are already in the data lake (the status query is
  commented out), so a rerun appends everything again; merge_era5_conus.py keeps the first of
  any duplicated (sub-basin, date) rows.
  A failed batch is only printed; the script still exits normally. Check the log for
  "failed" before merging.
"""
from swotgat.paths import NAS_ICE, WORK
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from tqdm import tqdm
import rasterio.features
from rasterio.transform import from_origin
from shapely.geometry import box
from shapely.ops import transform
import pyproj
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import dask.config
from dask.distributed import Client, as_completed
from dask_jobqueue import SLURMCluster
from sklearn.cluster import KMeans
from data import BasinDataLake

ERA5_DIR = Path(f"{NAS_ICE}/data/ERA5-Land/")


def create_spatial_batches(gdf, batch_size):
    """Group the sub-basins of one basin into spatially compact batches with K-Means clustering.

    The number of batches is ceil(number of sub-basins / batch_size), so batch_size is an average,
    not a limit. Returns a list of dictionaries, one per batch, with the sub-basin ids ("ids"),
    their polygons ("geoms") and one bounding box for the whole batch ("bounds").
    Adds a "cluster" column to gdf.
    """
    n_batches = int(np.ceil(len(gdf)/batch_size))
        
    # Centroids for clustering. The projection is there mostly to avoid the warning about
    # centroids in geographic coordinates; these centroids do not need to be accurate.
    gdf_proj = gdf.to_crs("EPSG:3857")
    centroids = np.array([list(p.coords)[0] for p in gdf_proj.geometry.centroid])
    
    # Perform K-Means clustering
    kmeans = KMeans(n_clusters=n_batches, random_state=42, n_init=10)
    gdf['cluster'] = kmeans.fit_predict(centroids)
    
    batches = []
    for i in range(n_batches):
        batch_gdf = gdf[gdf['cluster'] == i]
        if batch_gdf.empty:
            continue
        
        # Ids of all sub-basins in this cluster
        batch_ids = batch_gdf.index.tolist()
        
        # Calculate a single, unified bounding box for the entire batch
        # Add a 0.1 degree padding to ensure all grid cells are included
        min_lon, min_lat, max_lon, max_lat = batch_gdf.total_bounds
        padding = 0.1
        batch_bounds = (
            np.floor((min_lon - padding) * 10) / 10,
            np.floor((min_lat - padding) * 10) / 10,
            np.ceil((max_lon + padding) * 10) / 10,
            np.ceil((max_lat + padding) * 10) / 10,
        )
        
        batches.append({
            "ids": batch_ids,
            "geoms": batch_gdf.geometry,
            "bounds": batch_bounds
        })
        
    return batches


def open_monthly_files(year, month):
    """Open the two ERA5-Land files of one month (accumulated and averaged variables), merged.

    Returns one xarray Dataset holding the variables of both files.
    """
    fps = [
        ERA5_DIR / "accumulated" / f"{year}_{month:02d}.nc",
        ERA5_DIR / "average" / f"{year}_{month:02d}.nc"
    ]
    datasets = [xr.open_dataset(fp, engine="h5netcdf") for fp in fps]
    ds = xr.merge(datasets, compat="override").drop_vars(['expver','number'], errors='ignore')
    return ds


def subset_ds_by_bounds(ds, bounds):
    """Cut a dataset to the box bounds = (min_lon, min_lat, max_lon, max_lat).

    Latitude is stored descending, so the latitude slice runs from max_lat to min_lat.
    """
    min_lon, min_lat, max_lon, max_lat = bounds
    return ds.sel(latitude=slice(max_lat, min_lat), longitude=slice(min_lon, max_lon))


def get_weight_matrix(ds_subset, watershed_geom):
    """Compute the exact coverage fractions of one sub-basin polygon (slow; used as the fallback).

    For every grid cell of ds_subset, the fraction is the area of the polygon-cell intersection
    divided by the cell area, both measured in EPSG:3857. Cells are 0.1 x 0.1 degree boxes
    centered on the grid coordinates. watershed_geom is a one-row GeoDataFrame.
    Returns a DataArray of fractions on (latitude, longitude).
    """
    lat, lon = ds_subset['latitude'].values, ds_subset['longitude'].values
    lat_res, lon_res = 0.1, 0.1
    
    watershed_proj = watershed_geom.to_crs(epsg=3857)
    transformer = pyproj.Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
    
    coverage_fractions = np.zeros((len(lat), len(lon)))
    
    for i, lat_val in enumerate(lat):
        for j, lon_val in enumerate(lon):
            pixel_box = box(lon_val - lon_res/2, lat_val - lat_res/2,
                            lon_val + lon_res/2, lat_val + lat_res/2)
            pixel_box_proj = transform(transformer.transform, pixel_box)
            
            intersection_area = watershed_proj.intersection(pixel_box_proj).area
            pixel_area = pixel_box_proj.area
            coverage_fractions[i, j] = (intersection_area / pixel_area).iloc[0] if pixel_area > 0 else 0
    
    return xr.DataArray(
        coverage_fractions,
        dims=["latitude", "longitude"],
        coords={"latitude": lat, "longitude": lon}
    )


def get_weight_matrix_fast(ds_subset, watershed_geom, scale_factor=10):
    """Compute the coverage fraction of each grid cell by one sub-basin polygon, by rasterization.

    The polygon is rasterized on a grid scale_factor times finer than the data grid (a fine cell
    counts when its center lies inside the polygon), and the fine mask is averaged back to the
    data grid.

    Args:
        ds_subset: xarray Dataset/DataArray with 1-D latitude/longitude coordinates
        watershed_geom: one-row GeoDataFrame with the sub-basin polygon
        scale_factor: fine cells per data cell along each axis (10 gives fractions in steps
            of 1/100)

    Returns:
        DataArray named "weights" on (latitude, longitude), with the fraction of each cell
        covered by the polygon.
    """
    # 1. Extract coordinates and resolution
    # Assumes 1D coordinates for lat/lon
    lat = ds_subset['latitude'].values
    lon = ds_subset['longitude'].values
    
    # Calculate resolution (assuming uniform grid)
    # ERA5 lat is usually descending, so d_lat might be negative
    d_lat = lat[1] - lat[0]
    d_lon = lon[1] - lon[0]
    
    # 2. Define Coarse Dimensions
    height = len(lat)
    width = len(lon)
    
    # 3. Define Fine Dimensions (Super-sampling)
    fine_height = height * scale_factor
    fine_width = width * scale_factor
    
    # 4. Create Affine Transform for the FINE grid
    # We shift the origin to the top-left corner of the first pixel
    # Note: We subtract/add half a pixel to move from center to corner
    west = lon[0] - d_lon / 2
    north = lat[0] - d_lat / 2
    
    # The fine resolution is the coarse resolution divided by the scale factor.
    # from_origin's xsize/ysize must be positive -- it negates ysize internally
    # to build a north-up transform, so passing a signed d_lat double-negates it.
    fine_transform = from_origin(
        west, 
        north, 
        abs(d_lon) / scale_factor, 
        abs(d_lat) / scale_factor
    )
    
    # 5. Rasterize onto the fine grid (binary 0/1)
    # rasterio expects the geometry coordinates to match the transform coordinates
    # (longitude/latitude here), so the polygon is converted to EPSG:4326 if needed.
    if watershed_geom.crs != "EPSG:4326":
        watershed_geom = watershed_geom.to_crs("EPSG:4326")
        
    mask_fine = rasterio.features.rasterize(
        shapes=[(watershed_geom.geometry.iloc[0], 1)],
        out_shape=(fine_height, fine_width),
        transform=fine_transform,
        fill=0,
        all_touched=False,  # a fine cell counts only if its center is inside the polygon
        dtype=np.float32
    )
    
    # 6. Aggregate (downsample) to coarse grid
    # Reshape to (Coarse_H, Factor, Coarse_W, Factor)
    # Then take the mean over the two 'Factor' axes
    mask_coarse = mask_fine.reshape(
        height, scale_factor, width, scale_factor
    ).mean(axis=(1, 3))
    
    # 7. Return as DataArray matching input ds
    return xr.DataArray(
        mask_coarse,
        dims=("latitude", "longitude"),
        coords={"latitude": lat, "longitude": lon},
        name="weights"
    )


def get_results_df(ds_sub, weights):
    """Compute the coverage-weighted mean and variance of every variable at every time step.

    The statistics are taken over the grid cells of ds_sub with the coverage fractions in
    weights. Uses xarray's weighted mean and variance, which leave NaN cells out of both the sum
    and the weight total. Returns a DataFrame with columns <variable>_mean and <variable>_var.
    """
    dims = ["latitude", "longitude"]
    ds_sub_w = ds_sub.weighted(weights)
    
    mean_df = (ds_sub_w.mean(dim=dims).to_dataframe()
               .rename(lambda n: n+"_mean", axis=1))
    var_df = (ds_sub_w.var(dim=dims).to_dataframe()
               .rename(lambda n: n+"_var", axis=1))
    
    return pd.concat([mean_df, var_df], axis=1)


def process_spatial_batch(spatial_batch: dict, start_date, end_date):
    """Extract the forcing of one spatial batch of sub-basins from start_date to end_date.

    The coverage weights of every sub-basin are computed once. Each monthly file is then opened
    once, cut to the batch bounding box and applied to all sub-basins of the batch.
    Returns {sub-basin id: DataFrame of all months}. In the comments below, "COMID" means the
    sub-basin id.
    """
    batch_ids = spatial_batch['ids']
    batch_geoms = gpd.GeoDataFrame(spatial_batch['geoms'])
    batch_bounds = spatial_batch['bounds']

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    
    # Compute the weight matrices of all sub-basins in the batch once.
    # The first month's file supplies the grid.
    print(f"Pre-calculating weight matrices for batch...")
    sample_ds = open_monthly_files(start_date.year, start_date.month)
    sample_ds_sub = subset_ds_by_bounds(sample_ds, batch_bounds)
    
    weights_map = {}
    n_fallback = 0
    for comid, row in batch_geoms.iterrows():
        watershed = gpd.GeoDataFrame([row], crs=batch_geoms.crs)
        weights = get_weight_matrix_fast(sample_ds_sub, watershed)
        # The super-sampled raster (see get_weight_matrix_fast) can miss a sub-basin smaller than
        # about one fine grid cell entirely. Its coverage is then zero everywhere, which would
        # silently give NaN in the weighted mean below. For these, fall back to the slow, exact
        # polygon-intersection method. In practice this is a small share of sub-basins (the
        # smallest slivers).
        if float(weights.sum()) <= 0:
            weights = get_weight_matrix(sample_ds_sub, watershed)
            n_fallback += 1
        weights_map[comid] = weights
    sample_ds.close()
    print(f"Weight matrices calculated ({n_fallback} fell back to exact method for near-zero coverage).")

    # Generate date range
    date_range = pd.date_range(start=start_date, end=end_date, freq="MS")
    
    # This dictionary will hold the time-series data for each COMID
    comid_results = {comid: [] for comid in batch_ids}

    # Outer loop is TIME
    for date in date_range:
        # Load the full monthly file and subset to the BATCH's bounds ONCE
        ds_full = open_monthly_files(date.year, date.month)
        ds_region = subset_ds_by_bounds(ds_full, batch_bounds)
        
        # Inner loop is COMID (processing from in-memory data)
        for comid in batch_ids:
            weights = weights_map[comid]
            
            # Ensure weights align with the regional data subset
            weights_aligned = weights.reindex_like(ds_region, method='nearest', tolerance=0.01)
            
            df_month = get_results_df(ds_region, weights_aligned)
            comid_results[comid].append(df_month)
        
        ds_full.close()
        ds_region.close()

    cat_dfs = {}
    for comid, df_list in comid_results.items():
        cat_dfs[comid] = pd.concat(df_list, axis=0)

    return cat_dfs


def process_and_write_batch(basin_id, batch_dict, start_date, end_date, save_dir):
    """Extract one spatial batch and append it to the data lake (the task run on a Dask worker).

    batch_dict is one batch from create_spatial_batches; save_dir is the BasinDataLake folder.
    Returns the number of sub-basins written.
    """
    # Process the batch to get the dataframes in memory (on the worker)
    subbasin_df_dict = process_spatial_batch(batch_dict, start_date, end_date)
    
    # Initialize the store connection
    store = BasinDataLake(save_dir)
    store.write_dynamic(basin_id, 'era5', subbasin_df_dict, mode='append')
    
    # Just return number of subbasins we finished
    return len(subbasin_df_dict)



if __name__ == "__main__":  
    parser = argparse.ArgumentParser()
    parser.add_argument("--basin-file", required=True,
                        help="Path to subbasin file directory")
    parser.add_argument("--save-dir", required=True,
                        help="Location of the deltalake dataset")
    parser.add_argument("--n-workers", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--start-date", default="1980-01-01")
    parser.add_argument("--end-date", default="2025-12-31")
    
    args = parser.parse_args()
    print("\n========== Program Arguments ==========")
    for k, v in vars(args).items():
        print(f"{k:>12}: {v}")
    print("=======================================\n")

    print ("Reading subbasin file ...")
    subbasins = gpd.read_parquet(args.basin_file)

    # Move the Mississippi basin to the top of the table. (The groupby below sorts by outlet_id
    # again, so this does not set the order in which batches are submitted.)
    target = 'USGS-07374000'
    subbasins = pd.concat([
        subbasins[subbasins["outlet_id"] == target], 
        subbasins[subbasins["outlet_id"] != target]
    ])

    print("Preparing subbasin batches ...")
    # All sub-basins are processed. The commented lines would skip those already in the data lake.
    store = BasinDataLake(args.save_dir)
    
    # status = store.get_processing_status(source='era5')
    # processed_subs = [] if status.empty else status.index.get_level_values('subbasin').unique()
    # to_process = subbasins[~subbasins.index.isin(processed_subs)]
    to_process = subbasins

    # Precalculate all the batches we need to iterate over.
    all_batches = []
    for basin_name, basin_gdf in to_process.groupby('outlet_id'):
        batches = create_spatial_batches(basin_gdf, args.batch_size)
        for b in batches:
            all_batches.append((basin_name, b))

    print("Starting dask cluster ...")
    # Longer timeouts, because the tasks hold the Python GIL for long stretches and workers
    # can be slow to answer the scheduler
    dask.config.set({
        "distributed.comm.timeouts.connect": "60s",  # Time to establish connection
        "distributed.comm.timeouts.tcp": "60s",      # Time to wait for a response
        "distributed.scheduler.worker-ttl": "120s",  # Scheduler waits this long before removing worker (Default 30s-60s)
    })
    
    cluster = SLURMCluster(
        job_name="era5-dask-worker",
        queue="cpu",
        cores=1,
        processes=1,
        memory="32GB",
        walltime="2-00:00:00",
        log_directory=Path.cwd() / "_dask_workers",
        job_script_prologue=[
            f"cd {WORK}/swot-ml",
            "source .venv/bin/activate",
        ],
    )
    
    try:
        cluster.scale(args.n_workers)
        client = Client(cluster)
        client.wait_for_workers(args.n_workers)

        # submission loop
        futures_dict = {}
        for basin_id, batch_dict in all_batches:
            future = client.submit(
                process_and_write_batch,
                basin_id,
                batch_dict,
                args.start_date,
                args.end_date,
                args.save_dir
            )
            futures_dict[future] = basin_id

        # results loop
        with tqdm(total=len(futures_dict), desc="Processing spatial batches") as pbar:
            for future in as_completed(futures_dict):
                basin_id = futures_dict[future]
                try:
                    num_processed = future.result()
                    print(f"Batch for {basin_id} completed, processed {num_processed} subbasins.")
                except Exception as e:
                    print(f"batch for {basin_id} failed: {e}")
                finally:
                    pbar.update(1)

    finally:
        if 'client' in locals() and client.status != 'closed':
            client.close()
        if 'cluster' in locals() and cluster.status != 'closed':
            cluster.close()