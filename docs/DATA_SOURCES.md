# Data sources

Every raw dataset the pipeline reads, with the version used and where the
scripts expect it. `WORK`, `NAS_ICE` and `NAS_WATER` are the roots set in
`config/paths.env`. Entries marked **[confirm]** need a check by the authors
before publication (version, DOI or redistribution terms).

## Stages 01-02: graph and static attributes

| Dataset | Version, citation, access | Expected at |
|---|---|---|
| MERIT Hydro (flow direction, upstream cells, upstream area) | v1.0.1; Yamazaki et al. (2019), doi:10.1029/2019WR024873; http://hydro.iis.u-tokyo.ac.jp/~yamadai/MERIT_Hydro/ | `NAS_ICE/data/MERIT-Hydro/v1.0.1` |
| MERIT-Basins (catchments, river segments; the v0.7 ↔ bugfix1 segment table) | v0.7 / v1.0 bugfix1; Lin et al. (2019), doi:10.1029/2019WR025287; https://www.reachhydro.org/home/params/merit-basins | `NAS_ICE/data/MERIT_Basins/MERIT_Hydro_v07_Basins_v01_bugfix/`, `NAS_ICE/data/MERIT_Basins/v07_to_bugfix/` |
| SWOT Prior Lake Database, reservoir subset | v1.06 **[confirm source and citation]** | `WORK/LakeSPvC_ts_for_Ted/SWOT_PLD_v106_reservoirs.gdb`; some scripts read the identical copy at `NAS_WATER/cjgleason/ted/LakeSPvC_ts_for_Ted/` |
| Gauge-to-outlet matchups | shipped: `vendor/graph_delineator/all_matchups.csv` | — |
| USGS site table (drainage areas) | USGS National Water Information System, site service, downloaded with github.com/tedlanghorst/global_gauges | `WORK/global_gauges_data/usgs/site_metadata/data.parquet` |
| BasinATLAS | v1.0, level 9; Linke et al. (2019), doi:10.6084/m9.figshare.9890531 | `NAS_ICE/data/HydroSHEDS/HydroATLAS/BasinATLAS_v10.gdb` |
| Global Dam Watch | v1.0; Lehner et al. (2024), doi:10.6084/m9.figshare.25988293 | `WORK/GDW_v1_0/GDW_v1_0_shp/`, `NAS_WATER/cjgleason/ted/GDW/GDW_v1_0_shp/` |

## Stage 03: forcing

| Dataset | Version, citation, access | Expected at |
|---|---|---|
| ERA5-Land hourly (accumulated variables at 00:00: ssrd, strd, ro, sro, e, tp) | Muñoz-Sabater et al. (2021); Copernicus Climate Data Store `reanalysis-era5-land`, doi:10.24381/cds.e2161bac | `NAS_ICE/data/ERA5-Land/accumulated/YYYY_MM.nc` (1980-2025) |
| ERA5-Land daily means (sp, d2m, t2m, fal) | CDS `derived-era5-land-daily-statistics`, doi:10.24381/cds.e9c9c792 | `NAS_ICE/data/ERA5-Land/average/YYYY_MM.nc` (1980-2025) |
| ERA5-Land January-September 2026 | same products, downloaded 14-15 October 2026 by `03_forcings/c_extension_2026/download_2026.py`; for March, June and September the daily means were computed from the 00/06/12/18 UTC fields (`download_inst.py`, `prep_inst.py`) because the daily-statistics queue stalled | `WORK/CONUS_storage_fork/eval/overnight/era5_2026/raw/` |

A CDS download needs your own key in `~/.cdsapirc` and acceptance of the ERA5-Land licence. No key is stored in this repository.

## Stage 04: SWOT targets

| Dataset | Version, citation, access | Expected at |
|---|---|---|
| SWOT Level 2 lake product, prior lakes | `SWOT_L2_HR_LakeSP_prior_D` (version D), July 2023 to 10 October 2026, and `SWOT_L2_HR_LakeSP_prior_2.0` (version C) for the version offset; PO.DAAC Hydrocron API v1, https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1/timeseries (no login) **[confirm product DOIs]** | downloaded by `hc_pull.py` to `WORK/CONUS_storage_fork/eval/swotext/hc_all_v{C,D}.parquet` |
| SWOT Level 2 river product, reaches | `SWOT_L2_HR_RiverSP` version D: archive copy to 28 June 2026, then Hydrocron (`SWOT_L2_HR_RiverSP_D`) to 10 October 2026 **[confirm product DOIs]** | `NAS_ICE/data/SWOT/RiverSP_Reach_vD/parquet/NA.parquet`; `hc_reach_vD_2026.parquet` |
| HALF v1.0 (lake outlier filter) | Wang & Trudel, doi:10.5281/zenodo.20782248; Trudel et al. (in review) | shipped: `vendor/half_v1` |
| LakeSP version-C daily series with storage curves (static-area, SWOT area-elevation, 3D-LAKES) | prepared by Jida Wang's group; **unpublished [confirm citation and whether the curve parameters may be archived]** | `NAS_WATER/cjgleason/ted/LakeSPvC_ts_for_Ted/daily_ts_from_LakeSPvC/` |
| MERIT-Basins → SWORD reach table ("MHS") | prepared by Heejin An **[confirm citation]** | `NAS_WATER/cjgleason/ted/swot-ml/data/Heejin_MHS/` |
| SWOT Validation Set (gauge-to-reach matches) | v1.1.2 (SWORD v17c) **[confirm citation]** | `WORK/SVS/SVS_v1_1_2.nc` |
| Global gauge list | from the upstream model repository's data preparation | `WORK/Columbia_testing/swot-ml-data/Columbia_05km2/metadata/gauges.parquet` |

## Stage 07: evaluation

| Dataset | Version, citation, access | Expected at |
|---|---|---|
| ResOpsUS (reservoir storage, inflow, outflow to 2020) | Steyaert et al. (2022), doi:10.1038/s41597-022-01134-7 **[confirm the Zenodo record and version used]** | `WORK/CONUS_storage_fork/ResOpsUS/` |
| Agency reservoir storage, 2023-2026 | downloaded September-November 2026 from: USACE CWMS Data API; USGS NWIS; NRCS AWDB; Reclamation RISE, HDB and Hydromet; Texas Water Development Board; California Data Exchange Center; Colorado Division of Water Resources | caches in `WORK/CONUS_storage_fork/agency_cache/`, `gauges2026/cache/`, `gauges_expand/cache/` |
| GRADES-hydroDL river discharge | V2.0, Pfafstetter region 7, 1980-01-01 to 2026-06-12; Yang et al. (2025) **[confirm citation and the public folder link]**; file ids in `07_evaluation/b_rule_models/inputs/gdrive_listing.json` | `WORK/GRADES/qout/` |
| GDROM rules and cleaned records | version 2, HydroShare **[confirm citation and resource id]** | `WORK/GRADES/gdrom_v2/` |
| MODROM calibrated parameters | Li & Villarini (2026), HydroShare **[confirm]** | `WORK/GRADES/modrom/` |
| ISTARF-CONUS rule parameters | Turner et al. (2021), doi:10.1016/j.jhydrol.2021.126843 **[confirm data DOI]** | `WORK/GRADES/ISTARF-CONUS.csv` |
| Statewide monthly Palmer Drought Severity Index | NOAA NCEI Climate at a Glance, 1979-2026 | `WORK/GRADES/pdsi_state_monthly_1979_2026.csv` |

## Stage 08: figures

| Dataset | Version, access | Expected at |
|---|---|---|
| North American Level I ecoregions | Commission for Environmental Cooperation / US EPA, `NA_CEC_Eco_Level1` | reduced to `08_figures/inputs/eco_l1_*.{csv,json}` (shipped) **[the script that made these three files was not found]** |
| CONUS outline, US state outlines | **[source not recorded]** | `08_figures/inputs/conus_outline_albers.json`, `us_states_rings.json` (shipped) |
