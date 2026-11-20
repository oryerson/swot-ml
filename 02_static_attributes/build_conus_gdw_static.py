"""Add 13 dam columns from Global Dam Watch (GDW) to the static-attribute table.

Each lake node is matched to a GDW reservoir by polygon overlap and receives that
dam's capacity, height, depth, degree of regulation and uses. Lake nodes without
a GDW match get imputed values. All other nodes get zeros (free-flowing).

Reads (file names come from three environment variables)
  $V35S_SUBS                                   sub-basin table (GeoParquet) with is_reservoir and uparea_km2
  $V35S_WORK/static_conus_$V35S_TAG.parquet    static table from build_conus_static.py
  $SWOTGAT_WORK/GDW_v1_0/GDW_v1_0_shp/GDW_reservoirs_v1_0.shp        GDW reservoir polygons
  $SWOTGAT_WORK/LakeSPvC_ts_for_Ted/SWOT_PLD_v106_reservoirs.gdb     lake outlines of the SWOT Prior Lake
                                               Database (PLD), for the lake surface area used in imputation

Writes
  $V35S_WORK/static_conus_gdw_$V35S_TAG.parquet      the static table plus the 13 columns
  $V35S_WORK/gdw_match_report_$V35S_TAG.csv          the matched dams (name, main use, capacity, height, DOR)

Usage
  V35S_SUBS=... V35S_WORK=... V35S_TAG=v35 python build_conus_gdw_static.py [--impute area|median|zero]

Columns added
  is_reservoir        1 for a lake node, else 0
  dam_has_gdw         1 if the lake node has a GDW match, else 0; marks imputed rows so the model can discount them
  dam_cap_mcm         storage capacity (million m3; GDW CAP_MCM)
  dam_hgt_m           dam height (m; GDW DAM_HGT_M)
  dam_depth_m         mean depth (m; GDW DEPTH_M)
  dam_dor_pc          degree of regulation (%; GDW DOR_PC)
  dam_cap_per_uparea  dam_cap_mcm / uparea_km2 of the node
  use_elec, use_irri, use_fcon, use_supp, use_recr, use_navi
                      GDW use flags as an ordinal: Major = 3, Main = 2, Sec = 1, otherwise 0

Method
  Match: a lake node's polygon is joined to every GDW reservoir polygon it intersects; if there are several, the
  one with the largest capacity is kept. (fix_gdw_static.py later replaces this match by a stricter one.)
  No-data: GDW's -99 is turned into missing for capacity, height, depth and DOR, and so is a height <= 0.
  Imputation for lake nodes with missing values (--impute):
    area    (default) capacity from the lake's PLD surface area A: a straight-line fit of log10(capacity) on
            log10(A) over the matched lakes (needs at least 50 of them, else falls back to median). Depth is the
            imputed capacity divided by A (1 million m3 per km2 is 1 m). Height and DOR, which have no basis in
            area, get the median of the matched lakes. A capacity or depth that is still missing gets the median
            of that column over all lake nodes that have a value by then.
    median  every missing value gets the column's median over the lake nodes that have a value
    zero    every missing value gets 0 (asserts that there is no dam)
"""
from swotgat.paths import WORK
import argparse
import os as _os  # file names come from the environment: V35S_SUBS, V35S_WORK, V35S_TAG
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

warnings.filterwarnings("ignore")

REPO       = Path(f"{WORK}")
SAVE_DIR   = REPO / "CONUS"
SUBBASINS  = Path(_os.environ["V35S_SUBS"])
STATIC_IN  = Path(_os.environ["V35S_WORK"]) / f"static_conus_{_os.environ['V35S_TAG']}.parquet"
GDW_RES    = REPO / "GDW_v1_0/GDW_v1_0_shp/GDW_reservoirs_v1_0.shp"
STATIC_OUT = Path(_os.environ["V35S_WORK"]) / f"static_conus_gdw_{_os.environ['V35S_TAG']}.parquet"
REPORT_OUT = Path(_os.environ["V35S_WORK"]) / f"gdw_match_report_{_os.environ['V35S_TAG']}.csv"

NEW_NUMERIC = ["dam_cap_mcm", "dam_hgt_m", "dam_depth_m", "dam_dor_pc", "dam_cap_per_uparea"]
USE_COLS    = {"use_elec": "USE_ELEC", "use_irri": "USE_IRRI", "use_fcon": "USE_FCON",
               "use_supp": "USE_SUPP", "use_recr": "USE_RECR", "use_navi": "USE_NAVI"}
USE_ORDINAL = {"Major": 3, "Main": 2, "Sec": 1}


PLD_GDB = REPO / "LakeSPvC_ts_for_Ted/SWOT_PLD_v106_reservoirs.gdb"


def parse_args():
    """Parse the command-line arguments."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--impute", choices=["area", "median", "zero"], default="area",
                    help="How to fill dam attributes for reservoir nodes with no GDW match "
                         "(37%% of nodes at CONUS scale). "
                         "'area' (default) predicts capacity from the node's SWOT-PLD surface "
                         "area using a log-log fit to the matched reservoirs; "
                         "'median' assigns the matched-set median to every node (the original "
                         "behaviour, kept for reproducibility); "
                         "'zero' asserts no dam.")
    return ap.parse_args()


def pld_areas(res_ids, bbox):
    """Return the surface area (km2) of each reservoir node's SWOT-PLD water body.

    res_ids are the lake node ids and bbox the lon/lat box in which the PLD outlines are read. The area is
    computed from the outline in the equal-area projection EPSG:6933. Return a Series indexed by lake id;
    lakes without an outline in the box are absent.
    """
    pld = gpd.read_file(PLD_GDB, bbox=bbox)
    idc = [c for c in pld.columns if c.lower() in ("lake_id", "lakeid")][0]
    pld[idc] = pld[idc].astype("int64").astype(str)
    pld = pld[pld[idc].isin(set(res_ids))]
    return (pld.set_geometry(pld.geometry).to_crs("EPSG:6933")
              .assign(area_km2=lambda d: d.geometry.area / 1e6)
              .set_index(idc)["area_km2"])


def main():
    """Match lake nodes to GDW reservoirs, impute the missing values, and write the table and the match report.

    The imputation mode is read from the attribute main._impute_mode (set from --impute; default "area").
    """
    subs = gpd.read_parquet(SUBBASINS).to_crs(4326)
    subs.index = subs.index.astype(str)
    res = subs[subs["is_reservoir"]].copy()
    print(f"Reservoir nodes: {len(res)}")

    gdw_cols = ["GDW_ID", "DAM_NAME", "MAIN_USE", "CAP_MCM", "DAM_HGT_M",
                "DEPTH_M", "DOR_PC", *USE_COLS.values(), "geometry"]
    gdw = gpd.read_file(GDW_RES, bbox=tuple(res.total_bounds))[gdw_cols].to_crs(4326)
    print(f"GDW reservoirs in basin bbox: {len(gdw)}")

    j = gpd.sjoin(res[["uparea_km2", "geometry"]].reset_index(names="res"), gdw,
                  how="left", predicate="intersects")
    j = j.sort_values("CAP_MCM", ascending=False).drop_duplicates("res")
    j = j.set_index("res").reindex(res.index)
    matched = j["GDW_ID"].notna()
    print(f"Reservoir nodes matched to a GDW dam: {matched.sum()}/{len(res)}")

    # GDW uses -99 as a no-data value. Left in place it would reach the static table, and
    # because dam_dor_pc is log-normalized, log1p(-99) is NaN: the model would get a NaN
    # input and the loss would be NaN from the first step. About 6 of 3,466 CONUS
    # reservoirs carry it.
    for _c in ["CAP_MCM", "DAM_HGT_M", "DEPTH_M", "DOR_PC"]:
        j.loc[j[_c] <= -99, _c] = np.nan
    j.loc[j["DAM_HGT_M"] <= 0, "DAM_HGT_M"] = np.nan

    out = pd.DataFrame(index=res.index)
    out["is_reservoir"] = 1
    out["dam_has_gdw"]  = matched.astype("int16").values
    out["dam_cap_mcm"]  = j["CAP_MCM"].values
    out["dam_hgt_m"]    = j["DAM_HGT_M"].values
    out["dam_depth_m"]  = j["DEPTH_M"].values
    out["dam_dor_pc"]   = j["DOR_PC"].values
    for new, src in USE_COLS.items():
        out[new] = j[src].map(USE_ORDINAL).fillna(0).values.astype("int16")

    mode = getattr(main, "_impute_mode", "area")
    miss = out["dam_cap_mcm"].isna()
    print(f"  imputation mode: {mode}  ({int(miss.sum())} unmatched of {len(out)})")

    if mode == "area" and miss.any():
        # Capacity is roughly surface area times depth, so fit log10(capacity) on
        # log10(PLD area) over the matched reservoirs and apply the fit to the rest.
        # Unlike a constant, this scales with an observed quantity, and it avoids
        # asserting either "median dam" or "no dam" for water bodies that GDW does not
        # list. GDW under-represents small dams, so the unmatched set is a mix of real
        # unlisted dams and natural lakes that cannot be told apart by geometry.
        # Fit quality on CONUS: r = 0.705, R^2 = 0.50, residual scatter about 3.75x (1 sigma).
        areas = pld_areas(out.index, tuple(res.total_bounds))
        a_all = areas.reindex(out.index)
        fit_ok = (~miss) & a_all.notna() & (out["dam_cap_mcm"] > 0) & (a_all > 0)
        if fit_ok.sum() >= 50:
            x = np.log10(a_all[fit_ok].values); y = np.log10(out.loc[fit_ok, "dam_cap_mcm"].values)
            slope, icept = np.polyfit(x, y, 1)
            r = float(np.corrcoef(x, y)[0, 1])
            print(f"    log-log fit: slope={slope:.3f} intercept={icept:.3f} r={r:.3f} (n={int(fit_ok.sum())})")
            est_ok = miss & a_all.notna() & (a_all > 0)
            est = 10 ** (slope * np.log10(a_all[est_ok].values) + icept)
            out.loc[est_ok, "dam_cap_mcm"] = est
            # implied mean depth follows from the same fit (1 MCM / 1 km2 == 1 m)
            out.loc[est_ok, "dam_depth_m"] = est / a_all[est_ok].values
            print(f"    area-imputed {int(est_ok.sum())}: median {np.median(est):.1f} MCM "
                  f"(vs constant {out.loc[~miss, 'dam_cap_mcm'].median():.1f})")
        else:
            print("    too few matched reservoirs to fit; falling back to median")
            mode = "median"

    if mode == "zero":
        for col in ["dam_cap_mcm", "dam_hgt_m", "dam_depth_m", "dam_dor_pc"]:
            out[col] = out[col].fillna(0.0).astype("float32")
    else:
        # Anything still missing (including height and DOR, which have no basis in area)
        # gets the median of the matched reservoirs. dam_has_gdw = 0 marks every one of
        # these rows, so the model can learn to discount them.
        for col in ["dam_cap_mcm", "dam_hgt_m", "dam_depth_m", "dam_dor_pc"]:
            n_missing = int(out[col].isna().sum())
            med = out[col].median()
            out[col] = out[col].fillna(med).astype("float32")
            if n_missing:
                print(f"  {col}: median-filled {n_missing} -> {med:.1f}")

    out = out.astype({c: "float32" for c in ["dam_cap_mcm", "dam_hgt_m", "dam_depth_m", "dam_dor_pc"]})
    with np.errstate(divide="ignore", invalid="ignore"):
        out["dam_cap_per_uparea"] = (out["dam_cap_mcm"] / res["uparea_km2"]).astype("float32")

    static = pd.read_parquet(STATIC_IN)              # MultiIndex (basin, subbasin)
    sub_ids = static.index.get_level_values(1).astype(str)
    aligned = out.reindex(sub_ids)
    aligned.index = static.index

    for col in NEW_NUMERIC:
        static[col] = aligned[col].fillna(0.0).astype("float32")
    for flag in ["is_reservoir", "dam_has_gdw", *USE_COLS]:
        static[flag] = aligned[flag].fillna(0).astype("int16")

    STATIC_OUT.parent.mkdir(parents=True, exist_ok=True)
    static.to_parquet(STATIC_OUT)
    new_feats = ["is_reservoir", "dam_has_gdw", *NEW_NUMERIC, *USE_COLS.keys()]
    print(f"\nWrote {STATIC_OUT}  shape={static.shape}  (+{len(new_feats)} cols)")

    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    report = j[matched][["DAM_NAME", "MAIN_USE", "CAP_MCM", "DAM_HGT_M", "DOR_PC"]].copy()
    report.index.name = "res_node_id"
    report.to_csv(REPORT_OUT)
    print(f"Match report ({len(report)} dams): {REPORT_OUT}")


if __name__ == "__main__":
    _a = parse_args()
    main._impute_mode = _a.impute
    main()
