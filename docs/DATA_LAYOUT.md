# Layout of the work tree

`$SWOTGAT_WORK` holds all data the pipeline writes, and (after
`python tools/deploy.py`) a copy of each script at the place it runs from.
The layout mirrors the project's original working folder, which is why some
folder names reflect the project's history rather than the pipeline order.
`provenance/files.tsv` (column `deploy_to`) gives the exact location of every
script; `bin/run --where <script>` prints it.

Main products, by stage (`W` = `$SWOTGAT_WORK`, `E` = `W/CONUS_storage_fork/eval`):

```
W/CONUS/                                    the model's inputs and runs
  basin_lists/basin_manifest.csv, gauges/   01  basins and their gauge tables
  merit_rasters/                            01  flow-direction and accumulation mosaics
  metadata/subbasins_conus_v35.parquet      01  sub-basins of graph v3.5            (md5 19a764d9)
  metadata/graphs_conus_v35.json            01  graphs                              (md5 c580577d)
  metadata/reach_area_conus_v35.csv         04b SWOT reach area per river node      (md5 0093a582)
  attributes/static_conus_storage_fixgdw_v35_g4.parquet   02  static attributes     (md5 a146e7dc)
  basin_lists/storage/{train,valid,test}_nodes_g_v35.csv  04d node lists
  basin_lists/fallback/dv_SWOTAE_vd2026_fb_g_v35.csv      04d lakes with a storage target
  basin_lists/swotqc/drop_river_paper_min_present_t9b_v35.parquet   04d river observation mask
  zarr_store/                               03a first store (2000-2025); source of the normalisation statistics
  zarr_store_ext1980/, zarr_store_vd2026/, zarr_store_vd2026_halfexact/   03b, 03c, 04c  template stores
  zarr_store_vd2026_halfexact_v35/          03d TRAINING STORE (forcing + SWOT targets, graph v3.5)
  era5_conus_v35c5_S{1,2}_*.json            05  the two model configurations (written by tools/deploy.py)
  era5_conus_v35c5_S1_storage_<time>/       05  run folder, seed 0: checkpoints/step_050000/{model.eqx,config.json,...}
  era5_conus_v35c5_S2_storage_s1_<time>/    05  run folder, seed 1

W/CONUS_storage_fork/
  features/longmem_v35/                     03e long-memory inputs 2009-2026
  predictions/<run>/step_050000_<tag>.parquet   06  SWOT-GAT predictions (tags h10, x26, x8085 ... x0409)
  predictions/grades_<model>3c_g_y1980/     07b rule-model runs 1980-2020
  predictions/grades_<model>3c_g26h0/       07b rule-model runs to June 2026
  ResOpsUS/, agency_cache/, catalogs/       07a raw gauge data and download caches
  crosswalk/                                07a dam-node and gauge-dam matching
  testsets/regime3_full_1980_2020/          07a frozen test set 1980-2020 (ResOpsUS)
  testsets/regime1x_2023_2026/              07a frozen test set, SWOT Period (agency gauges)
  gauges2026/, gauges_expand/               07a 2026 extension; expanded gauge set

E/graph_v3/                                 01  delineation_v34/, v34/, v35/ (graph builds), gauges_v34/, v35_work/
E/v35_inputs/statics/                       02  work/, final attribute table
E/v35_inputs/forcing/                       03d work/, zarr_store_ext1980_v35/, zarr_store_vd2026_v35/ (PREDICTION STORES)
E/v35_inputs/lists/                         04d node lists and masks before installation into W/CONUS
E/overnight/era5ext/, era5_2026/            03b, 03c  extension work folders (raw 2026 downloads in era5_2026/raw)
E/swotext/                                  04a, 04b  SWOT downloads (hc_all_vD.parquet ...), HALF outputs, river series
E/half_vd_review/lakesp_vD_halfexact_graph.parquet   04a  the lake record after the HALF filter and coverage rule
E/v21_retrain/store/                        04c  lake target table
E/v35_launch/                               05  cache check, logs, node lists of the 2010-2020 and SWOT Period predictions
E/c5_paper/                                 06  1980-2026 long-memory store, prediction logs; 07c series_ip.npz
E/overlap50/score/, tables/, fig1_o50.json  07c scores with the 50% overlap rule; matched sets; paired tests
E/overlap50/fig3c_sd/                       07c training coverage and harmonic irregularity tables
E/perf_attrib/out/                          07c driver tables (vars.parquet, ops_8120.parquet, blocks.csv)
E/grl_figs/intermediate/c5_o50_20261113/score/   07c full-record scores, rule-model ensemble, Figure 1 picks
E/grl_figs/intermediate/c5_o50_20261113/kit_d/   08  figure tables, the four figures, num_c5d_fig*.json
E/swot_vs_gauge/                            07d SWOT against gauges on pass days
W/GRADES/                                   07b GRADES-hydroDL discharge, rule parameters, node maps
```

Raw data expected under the three roots is listed in `docs/DATA_SOURCES.md`.
`tests/smoke/inputs_eval.txt` lists the files that re-scoring and the figures
need (about 15 GB).
