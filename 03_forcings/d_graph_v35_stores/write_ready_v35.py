"""Write the READY markers: JSON files recording that the graph v3.5 inputs are complete and checked.

A marker lists the input files with their md5 checksums and the results of the acceptance checks.
In the original run, the scripts that launched training and prediction read these markers before
starting. Every md5 is recomputed from the files when the marker is written.

Usage (<v35_inputs> = $SWOTGAT_WORK/CONUS_storage_fork/eval/v35_inputs)
    python write_ready_v35.py train   -> <v35_inputs>/READY_v35.json
        Training inputs as installed in $SWOTGAT_WORK/CONUS with the suffix _v35: graph, sub-basin
        table, reach-area table, training store, static attributes, node lists, and the two
        configuration templates (seeds S1 and S2). Reads the data-loader check
        <v35_inputs>/loader_check_era5_conus_v35c1_S1_storage.TEMPLATE.json and stops if it lists
        any failure; also reads <v35_inputs>/forcing/checks.json (checks_v35.py).
    python write_ready_v35.py ext     -> <v35_inputs>/READY_v35_ext.json
        The two prediction stores (forcing 1980-01-01 to 2026-09-30, the input of the long-memory
        features), with the pass/fail result of every gate in <v35_inputs>/forcing/checks_pred.json
        (checks_pred_v35.py).
    python write_ready_v35.py pred    -> <v35_inputs>/READY_v35_pred.json
        Which store each prediction window reads: `h10` (the 2010-2020 predictions, which need
        forcing from 2009) reads zarr_store_ext1980_v35, and `x26` (January 2023 to September
        2026, which contains the SWOT Period) reads zarr_store_vd2026_v35. Needs
        READY_v35_ext.json.

Notes
    An existing marker is not deleted. It is moved to <v35_inputs>/superseded/ with its
    modification time in the name, and the new marker records its path and md5 under `supersedes`.
    Many text fields of the markers (counts of river nodes, notes on decisions, who wrote the
    marker) are fixed strings that describe the paper's run. They are not recomputed, so the
    script documents that run rather than a new one.
    The `train` marker reads the configuration templates named `v35c1` (C1, an earlier model
    configuration on graph v3.5) and compares river-node counts with `a180` (the store and graph
    of an earlier model configuration).
"""
from swotgat.paths import WORK
import hashlib
import json
import sys
import time
from pathlib import Path

R = Path(f"{WORK}"); C = R/"CONUS"; E = R/"CONUS_storage_fork/eval"; V = E/"v35_inputs"
what = sys.argv[1]


def md5(p):
    """Return the md5 hex digest of the file p (read in blocks of 4 MB)."""
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""): h.update(b)
    return h.hexdigest()


def store_md5(z):
    """Return one md5 digest for the Zarr store z.

    It is the md5 of the lines "<group>:<md5 of the group's zarr.json>", sorted by group. This is
    the same digest that make_config_v35.py records.
    """
    return hashlib.md5("".join(f"{g.name}:{md5(g/'zarr.json')}\n" for g in sorted(Path(z).iterdir()) if (g/"zarr.json").is_file()).encode()).hexdigest()


def f(p):
    """Return the path and the md5 digest of the file p as a dictionary."""
    return dict(path=str(p), md5=md5(p))


now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
if what == "train":
    out = V/"READY_v35.json"
    gate = json.load(open(V/"loader_check_era5_conus_v35c1_S1_storage.TEMPLATE.json"))
    assert not gate["fails"], gate["fails"]
    fc = json.load(open(V/"forcing/checks.json"))
    S1, S2 = V/"config/era5_conus_v35c1_S1_storage.TEMPLATE.json", V/"config/era5_conus_v35c1_S2_storage_s1.TEMPLATE.json"
    cfg = json.load(open(S1))
    d = dict(
        marker="READY_v35", written_utc=now, by="graph-v2-map session (Claude)",
        owen_ok='2026-10-22 ~01:00 UTC, in the map session: "Yes: once every check passes, write graph v3.5 and its inputs into CONUS/ as new *_v35 files (nothing overwritten)."',
        installed_in_conus=True, install_manifest=str(V/"INSTALL_v35.md"),
        graph=f(C/"metadata/graphs_conus_v35.json"), subbasins=f(C/"metadata/subbasins_conus_v35.parquet"),
        reach_area=f(C/"metadata/reach_area_conus_v35.csv"),
        store=dict(path=str(C/"zarr_store_vd2026_halfexact_v35"), groups_zarr_json_md5=store_md5(C/"zarr_store_vd2026_halfexact_v35"),
                   dates="2020-01-01..2026-09-30", groups=102, gates=str(V/"forcing/CHECKS.md")),
        statics=dict(**f(C/"attributes/static_conus_storage_fixgdw_v35_g4.parquet"), source="v35_inputs/statics/static_conus_storage_fixgdw_v35_g4_r2.parquet",
                     new_column="merit_uparea_km2 (log-normalised in the config)", gates=str(V/"statics/CHECKS.md")),
        lists=dict(train=f(cfg["train_basin_file"]), valid=f(cfg["valid_basin_file"]), test=f(cfg["test_basin_file"]),
                   fallback_dv_SWOTAE=f(cfg["target_basin_files"]["dv_SWOTAE"]), river_mask_t9b=f(cfg["target_drop_masks"]["dv_river"]),
                   rule="train = every node with any finite dv_river / dv_SWOTAE / dv_staticPLD 2023-01-01..2026-09-30 + every test and valid node; "
                        "fallback = every lake with data + every test reservoir; test lists score only (Owen: no spatial holdout)"),
        configs=dict(S1=f(S1), S2=f(S2), differ_only_in=["model_args.seed", "_v21_note"], cache_dir=cfg["cache_dir"],
                     diff_vs_a180=str(V/"config/config_diff_TEMPLATE.md")),
        gate=dict(script=str(V/"config/loader_check_v35.py"), result="PASSED", json=str(V/"loader_check_era5_conus_v35c1_S1_storage.TEMPLATE.json"),
                  counts=gate["counts"], cache=gate.get("cache"), checks={k: v for k, v in gate.items() if isinstance(v, dict) and "ok" in v}),
        dv_river_vs_a180=dict(
            store_nodes_with_data=dict(a180_store=4388, v35_store=4451, gone_ids=195, new_ids=258),
            store_obs=dict(a180_store=466461, v35_store=470344),
            gone_ids_obs_19072="18 nodes inherited by a new id with the same reach set; 133 now at new ids with their own reach set; 27 still carried by a retained id; "
                               "15 with reaches in no v3.5 crosswalk (1,164 obs); 2 at new ids with < 10 obs (169 obs)",
            supervised_after_mask_in_graph_basins=dict(a180_nodes=4293, v35_nodes=gate["counts"]["river_nodes_with_data"],
                                                       a180_obs=359791, v35_obs=gate["counts"]["river_obs"]),
            a180_reference_4372="checker A's 4,372 = A180 graph nodes with dv_river before the mask",
            basins_losing_over_1pct=["USGS-01205500 15.4%", "USGS-08041780 11.6%", "USGS-01463500 9.6%", "USGS-01059000 7.5%", "USGS-02135200 4.1%",
                                     "USGS-02226000 3.5%", "USGS-08030500 2.7%", "USGS-02428400 1.3%"]),
        t9=dict(t9_nodes=34, t9_rekey="all 34 identity", t9b_transferred=9, t9b_template=11,
                t9b_not_transferred=[dict(template_node="73004216", basin="USGS-01463500", owner_gauge="USGS-01432110", days=199),
                                     dict(template_node="73012080", basin="USGS-02469761", owner_gauge="USGS-02433496", days=65)],
                note="Owen 'Keep T9': the t9b mask is kept for C1/C2/C3; gate check f: 0 finite cached dv_river days at the 43 T9/T9b nodes"),
        notes=["The 2 basins excluded since v2.x (USGS-02246500, USGS-02298202) are still outside the graph (100 of 102 basins).",
               "Owen 2026-10-22: 3 small gauges inside a much bigger node removed (USGS-02312667, USGS-13210831, USGS-02264495); none was in any list.",
               "Statics: dam/use values frozen to the v2.3 match where the match is unchanged (avoids GDW imputation drift); merit_uparea_km2 = MERIT upa at "
               "each node's routed outlet, with 8 nodes overridden to the graph uparea (4 pass-through gauges, 4 river slivers).",
               "Open (Owen, via the hub): 9 reservoirs whose pour-point merit_uparea_km2 is outside 0.5-2x of the graph (Heidecke, off-stream, split fragments) kept by definition.",
               "Known production quirk copied bitwise: 7310084002 (Susquehanna 2-cell reservoir; production extraction differs in 2020/22/23)."])
elif what == "ext":
    out = V/"READY_v35_ext.json"
    cp = json.load(open(V/"forcing/checks_pred.json"))
    d = dict(marker="READY_v35_ext", written_utc=now, by="graph-v2-map session (Claude)",
             purpose="v3.5 forcing 1980-01-01..2026-09-30 for longmem_v35 (same pairing as build_longmem_vd2026.py: 1980-2025 from ext1980, 2026 from vd2026)",
             ext1980=dict(path=str(V/"forcing/zarr_store_ext1980_v35"), dates="1980-01-01..2025-12-31", groups=102,
                          groups_zarr_json_md5=store_md5(V/"forcing/zarr_store_ext1980_v35"), manifest=str(V/"forcing/MANIFEST_ext1980_v35_group_zarr_json.md5"),
                          gates={k: v.get("PASS") for k, v in cp["ext1980"].items() if isinstance(v, dict) and "PASS" in v}),
             vd2026=dict(path=str(V/"forcing/zarr_store_vd2026_v35"), dates="2020-01-01..2026-09-30", groups=102,
                         groups_zarr_json_md5=store_md5(V/"forcing/zarr_store_vd2026_v35"), manifest=str(V/"forcing/MANIFEST_vd2026_v35_group_zarr_json.md5"),
                         gates={k: v.get("PASS") for k, v in cp["vd2026"].items() if isinstance(v, dict) and "PASS" in v}),
             method="nodes with an identical polygon copied bitwise from CONUS/zarr_store_ext1980 / zarr_store_vd2026 (20,081); new or reshaped polygons "
                    "(4,416) re-extracted with the era5ext method; normalization_stats verbatim",
             all_pass=dict(ext1980=cp["ext1980"]["ALL_PASS"], vd2026=cp["vd2026"]["ALL_PASS"]),
             known_exceptions=cp["ext1980"]["d2"].get("known_exceptions"),
             accepted_exception="(recorded in the gate's KNOWN list since the re-run; checker B F2) ext1980 gate d2 (method check at 1,000 copied validation nodes) flagged ONE node, 73005469 (split_downstream, 36.9 km2, "
                                "Susquehanna USGS-01578310): the production 2000-2019 values differ from the era5ext method (<= 5.5 sd in sro). Production is "
                                "self-consistent there (ext1980 == vd2026 for 2020-25) and the node is copied bitwise, so v3.5 equals production; same basin "
                                "and class as the known training-store quirk 7310084002. 999 of 1,000 validation nodes agree to < 1e-4 sd.",
             not_in_these_stores="the 32 long-memory features: built by feature-screen-v2 into CSF/features/longmem_v35 from these two stores")
elif what == "pred":
    out = V/"READY_v35_pred.json"
    e = json.load(open(V/"READY_v35_ext.json"))
    d = dict(marker="READY_v35_pred", written_utc=now, by="graph-v2-map session (Claude)",
             h10=dict(store=e["ext1980"]["path"], dates=e["ext1980"]["dates"], groups_zarr_json_md5=e["ext1980"]["groups_zarr_json_md5"],
                      replaces="CONUS/zarr_store_ext1980 (hard-coded in eval/pred_v21.sh)", covers="the 2010-2020 hindcast with 365-day history (from 2009)"),
             x26=dict(store=e["vd2026"]["path"], dates=e["vd2026"]["dates"], groups_zarr_json_md5=e["vd2026"]["groups_zarr_json_md5"],
                      replaces="CONUS/zarr_store_vd2026 (hard-coded in eval/pred_v21.sh)", covers="2023-01..2026-09 with 365-day history (from 2022)"),
             graph=str(C/"metadata/graphs_conus_v35.json"),
             required="pred_v21.sh must take these paths (no v2 defaults for a v35 run): a v3.5 graph node missing from a store group loads as all-NaN input "
                      "with no error (checker A F1). Every v3.5 graph node is present in every group of both stores (gate a).",
             kept_outside_conus="not copied into CONUS/ (ext1980_v35 is 22.7 GB); read in place from v35_inputs/forcing")
else:
    sys.exit("train | ext | pred")
if out.exists():   # writing a marker again keeps the earlier one (never deleted) in v35_inputs/superseded/
    (V/"superseded").mkdir(exist_ok=True); prev = V/"superseded"/f"{out.stem}.{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime(out.stat().st_mtime))}.json"
    assert not prev.exists(), prev; out.rename(prev); d["supersedes"] = dict(path=str(prev), md5=md5(prev))
json.dump(d, open(out, "w"), indent=1)
print(out, md5(out))
