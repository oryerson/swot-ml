"""Record of how the graph v3.5 extraction scripts were derived from the earlier scripts.

This script is kept as documentation. It is not run in the release: the scripts it generated are
shipped as files, and it cannot be re-run as it stands (see Notes).

In the original run, each `*_v35.py` script below was produced from an earlier script by text
substitution. Every substitution is listed here as a pair (old text, new text), and the script
asserts that the old text occurs exactly once before replacing it. The generated script is
therefore the original plus a short, checked list of changes, mostly paths. Everything else
(weighting functions, NaN handling, float64 arithmetic, river filters, anomaly, reach-area
handling) is the original code.

    original (release location)                generated script             changes
    b_extension_1980_1999/weights.py           weights_v35.py               polygons = graph v3.5
                                                                            sub-basins; output in
                                                                            <forcing>/work
    b_extension_1980_1999/extract.py           extract_v35.py               weights and yearly files
                                                                            in <forcing>/work
    b_extension_1980_1999/extract.py           extract_pred_v35.py          the same in
                                                                            <forcing>/work_pred (row
                                                                            subset of the weights)
    c_extension_2026/prep_2026.py              prep_2026_v35.py             weights and output in
                                                                            <forcing>/work; the 2026
                                                                            downloads are read from
                                                                            era5_2026/raw; only the
                                                                            `extract` command
    04_swot_targets/b_rivers/river_build_v2.py river_build_v35.py           sub-basins = graph v3.5;
                                                                            outputs *_v35 in
                                                                            <forcing>/work/river; the
                                                                            SWOT river download is
                                                                            read from swotext; also
                                                                            writes the sub-basin to
                                                                            reach lookup it used
    04_swot_targets/b_rivers/river_build_v2.py river_build_v35ra.py         as river_build_v35.py but
                                                                            with the graph v3.5
                                                                            reach-area table; outputs
                                                                            *_v35ra; no lookup file

<forcing> = $SWOTGAT_WORK/CONUS_storage_fork/eval/v35_inputs/forcing. In the release the first four
generated scripts are in this folder and the two river scripts are in 04_swot_targets/b_rivers.

Reads (original run): the six source scripts at their deployed locations under
    $SWOTGAT_WORK/CONUS_storage_fork/eval (overnight/era5ext, overnight/era5_2026, swotext).
Writes (original run): <forcing>/scripts/<generated script>, each with a first comment line that
    names its source and summarizes the changes.

Notes
    The old texts contain the absolute paths that the source scripts had on the cluster. The
    release versions of those scripts build their paths from swotgat.paths instead, so the
    assertions would fail on them. For the same reason the generated scripts in the release are
    not byte-identical to what this script wrote: their paths were converted and their comments
    and docstrings rewritten, with no other change to the code.
"""
from swotgat.paths import WORK
from pathlib import Path
CSF = f"{WORK}/CONUS_storage_fork"; E = f"{CSF}/eval"; F = f"{E}/v35_inputs/forcing"
V35 = f"{E}/graph_v3/v35/subbasins_conus_v35.parquet"
OUTD = Path(F)/"scripts"
def stage(src, dst, reps, header):
    """Write the script dst as a copy of the script src with the text replacements reps applied.

    reps is a list of pairs (old text, new text). Each old text must occur exactly once in src
    (asserted). The copy starts with a comment line that names src and gives header, a summary
    of the changes.
    """
    t = Path(src).read_text()
    for a, b in reps:
        assert t.count(a) == 1, (src, a)
        t = t.replace(a, b)
    (OUTD/dst).write_text(f'# [v35 staged copy of {src}; changes: {header}]\n' + t)
    print("wrote", OUTD/dst)
stage(f"{E}/overnight/era5ext/weights.py", "weights_v35.py",
      [(f'OUT = "{E}/overnight/era5ext"', f'OUT = "{F}/work"'),
       (f'G = gpd.read_parquet("{WORK}/CONUS/metadata/subbasins_conus.parquet")', f'G = gpd.read_parquet("{V35}")')],
      "OUT=forcing/work, polygons=v3.5 subbasins")
stage(f"{E}/overnight/era5ext/extract.py", "extract_v35.py",
      [(f'OUT = "{E}/overnight/era5ext"', f'OUT = "{F}/work"')], "OUT=forcing/work")
stage(f"{E}/overnight/era5_2026/prep_2026.py", "prep_2026_v35.py",
      [(f'D = Path("{E}/overnight/era5_2026"); RAW = D/"raw"', f'D = Path("{F}/work"); RAW = Path("{E}/overnight/era5_2026/raw")'),
       (f'EXT = Path("{E}/overnight/era5ext")', f'EXT = Path("{F}/work")'),
       ('    cmd = sys.argv[1]\n', '    cmd = sys.argv[1]; assert cmd == "extract", "v35 copy: extract only (merge/validate write under era5_2026)"\n')],
      "D=forcing/work, RAW read from era5_2026/raw, weights from forcing/work, extract only")
stage(f"{E}/swotext/river_build_v2.py", "river_build_v35.py",
      [(f'OUT = "{E}/swotext"', f'OUT = "{F}/work/river"; SWX = "{E}/swotext"'),
       ('args = M.parse_args(); subs = gpd.read_parquet(args.subbasins)',
        f'args = M.parse_args(); from pathlib import Path as _P; args.subbasins = _P("{V35}"); subs = gpd.read_parquet(args.subbasins)'),
       ('merged = M.build_reach_lookup(args, subs)\n',
        'merged = M.build_reach_lookup(args, subs)\n'
        'merged.rename_axis("subbasin").reset_index()[["subbasin", "reach_id", "outlet_id"]].to_csv(f"{OUT}/reach_lookup_v35.csv", index=False)\n'),
       ('hc = pd.read_parquet(f"{OUT}/hc_reach_vD_2026.parquet")', 'hc = pd.read_parquet(f"{SWX}/hc_reach_vD_2026.parquet")'),
       ('R.to_parquet(f"{OUT}/riversp_vD_present_graph.parquet", index=False)', 'R.to_parquet(f"{OUT}/riversp_vD_present_graph_v35.parquet", index=False)'),
       ('M_.to_parquet(f"{OUT}/drop_river_paper_min_present.parquet", index=False)', 'M_.to_parquet(f"{OUT}/drop_river_paper_min_present_v35.parquet", index=False)')],
      "subbasins=v3.5, outputs forcing/work/river/*_v35 + reach_lookup_v35.csv, Hydrocron pull read from swotext")
# Second river build: the river targets use the graph v3.5 reach areas. It reads
# work/river/reach_area_conus_v35.csv and writes outputs named *_v35ra.
stage(f"{E}/swotext/river_build_v2.py", "river_build_v35ra.py",
      [(f'OUT = "{E}/swotext"', f'OUT = "{F}/work/river"; SWX = "{E}/swotext"'),
       ('args = M.parse_args(); subs = gpd.read_parquet(args.subbasins)',
        f'args = M.parse_args(); from pathlib import Path as _P; args.subbasins = _P("{V35}"); subs = gpd.read_parquet(args.subbasins)'),
       ('hc = pd.read_parquet(f"{OUT}/hc_reach_vD_2026.parquet")', 'hc = pd.read_parquet(f"{SWX}/hc_reach_vD_2026.parquet")'),
       ('areas = pd.read_csv(f"{C}/metadata/reach_area_conus.csv", dtype={"subbasin": str})', 'areas = pd.read_csv(f"{OUT}/reach_area_conus_v35.csv", dtype={"subbasin": str})'),
       ('R.to_parquet(f"{OUT}/riversp_vD_present_graph.parquet", index=False)', 'R.to_parquet(f"{OUT}/riversp_vD_present_graph_v35ra.parquet", index=False)'),
       ('M_.to_parquet(f"{OUT}/drop_river_paper_min_present.parquet", index=False)', 'M_.to_parquet(f"{OUT}/drop_river_paper_min_present_v35ra.parquet", index=False)')],
      "subbasins=v3.5, reach areas=forcing/work/river/reach_area_conus_v35.csv, outputs forcing/work/river/*_v35ra, Hydrocron pull read from swotext")
# For the prediction stores: the same extract.py, reading the row subset of the weights
# (work_pred/weights.npz) and writing work_pred/years.
stage(f"{E}/overnight/era5ext/extract.py", "extract_pred_v35.py",
      [(f'OUT = "{E}/overnight/era5ext"', f'OUT = "{F}/work_pred"')], "OUT=forcing/work_pred (row subset of the v3.5 weights: re-extract nodes + validation sample)")
