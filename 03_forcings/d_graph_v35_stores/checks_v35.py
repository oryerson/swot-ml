"""Acceptance checks ("gates") for the graph v3.5 training store.

The checks compare the new store with its template, the training store of the same period on the
earlier node set. Both stores are only read. A gate either passes or fails; `ALL_PASS` in the
output is true only if every gate passes and no structural problem is found.

Reads (paths relative to $SWOTGAT_WORK; <forcing> = CONUS_storage_fork/eval/v35_inputs/forcing)
    <forcing>/zarr_store_vd2026_halfexact_v35    the new store (build_store_v35.py)
    CONUS/zarr_store_vd2026_halfexact            the template store
    <forcing>/work/node_map.csv, weights.npz, subbasins.csv, years/*.npy, template_nan_scan.json,
        river/reach_area_v35_all_reach_nodes.csv           products of the scripts of this folder
    CONUS_storage_fork/eval/graph_v3/v35/graphs_conus_v35.json   graph v3.5 (its node list)
    CONUS/metadata/graphs_conus_v22.json         the graph of an earlier graph version (v2.2), for
                                                 the river-node counts
    CONUS_storage_fork/eval/overnight/era5ext/weights.npz, subbasins.csv   weights of the earlier
                                                 node set (b_extension_1980_1999/weights.py)
    CONUS_storage_fork/eval/overnight/era5_2026/susq_map.json    the two Susquehanna sub-basins
                                                 whose 2026 forcing was mapped (c_extension_2026)
    CONUS_storage_fork/eval/v21_retrain/store/lake_targets_vd2026_halfexact.parquet   the lakes
                                                 with a target (3,224 in the paper's run)

Writes
    <forcing>/checks.json                              all results (also printed)
    <forcing>/work/checks_d_all_same_id_extract.csv    gate (d), every candidate node
    <forcing>/work/checks_river_by_basin.csv           river observations per basin, new and template
    <forcing>/work/checks_river_new_vs_gone.csv        river series of new nodes against the removed
                                                       earlier nodes with the same reach set

Usage: python checks_v35.py        (no arguments; run after all build_store_v35.py jobs)

Gates
    structure  Same basin groups, dates and variables as the template. Same array metadata (data
               type, codecs, fill value; chunks of 365 days x all sub-basins). Group attributes
               (`normalization_stats`) identical. Metadata consolidated. No finite target value
               before 2023.
    (a)   Every graph v3.5 node is a column of its basin's group. The columns of a group are
          exactly the v3.5 sub-basin rows of the basin, in order, with no duplicates.
    (b)   All 10 ERA5-Land means are finite at every column and date. (The count of non-finite
          values in the template, from prescan_template.py, is reported next to it.)
    (c)   200 random copied graph nodes (seed 35) are bitwise equal to the template in all 13
          variables (10 forcing variables, dv_SWOTAE, dv_staticPLD, dv_river). In addition all
          copied nodes must be bitwise equal in the forcing and the two lake targets.
    (d)   Nodes that were extracted again but share an id with a template node: daily t2m and tp
          must correlate with the template above 0.95, and the means must be plausible (t2m mean
          within 250-305 K and within 3 K of the template; tp mean in (0, 0.02] m/day and between
          0.5 and 2 times the template). The gate is decided on 100 random graph nodes; the result
          for all such nodes is reported.
    (d2)  Method check. The new extraction over the v3.5 polygons, taken at the copied nodes, is
          compared with the template column (which for 2020-2025 comes from the production
          extraction and for 2026 from the c_extension_2026 extraction). Differences are
          expressed as max |difference| / standard deviation of the template values of the basin.
          Pass: the 2026 values are exactly equal, and no node differs by more than 1e-4 standard
          deviations except the nodes listed in KNOWN. The two Susquehanna sub-basins of
          susq_map.json are left out, because the template maps their 2026 forcing.
    (d3)  The weight rows of all copied nodes are equal to the rows of the earlier weights file
          (the weights behind the template's 2026 values).
    (e)   dv_SWOTAE and dv_staticPLD are bitwise equal to the template at every lake with a
          target, and the total numbers of finite values are unchanged.
    (f)   Store size is reported; the work file system (WORK_FS in swotgat.paths) must have at
          least 100 GB free.
    river Reported, not a gate: river nodes lost and gained against the template, observations
          per basin, and agreement of new nodes with the removed nodes they replace.

Notes
    KNOWN documents one node (7310084002, a reservoir covering two grid cells in the Susquehanna
    basin USGS-01578310) where the production extraction of 2020, 2022 and 2023 differs from the
    method used here. The difference already exists in the template, and the node is copied bit
    for bit like every node with an unchanged polygon.
"""
from swotgat.paths import WORK, WORK_FS
import json, os, shutil, subprocess, numpy as np, pandas as pd, xarray as xr, zarr
from pathlib import Path
ROOT = Path(f"{WORK}"); C = ROOT/"CONUS"; T = C/"zarr_store_vd2026_halfexact"; CSF = ROOT/"CONUS_storage_fork"
F = CSF/"eval/v35_inputs/forcing"; W = F/"work"; N = F/"zarr_store_vd2026_halfexact_v35"; V35 = CSF/"eval/graph_v3/v35"
EXT = CSF/"eval/overnight/era5ext"
VARS = [f"{v}_mean" for v in ["ssrd", "strd", "ro", "sro", "e", "tp", "sp", "d2m", "t2m", "fal"]]; TG = ["dv_SWOTAE", "dv_staticPLD", "dv_river"]
rng = np.random.default_rng(35)
NM = pd.read_csv(W/"node_map.csv", dtype={"subbasin": str, "outlet_id": str, "template_group": str}, keep_default_na=False).set_index("subbasin")
gnodes = [str(n["id"]) for n in json.load(open(V35/"graphs_conus_v35.json"))["nodes"]]; gset = set(gnodes)
basins = sorted(p.name for p in T.iterdir() if (p/"zarr.json").is_file())
R = {}; prob = []
def P(msg):
    """Record the problem msg in the list of problems and print it."""
    prob.append(msg); print("PROBLEM", msg, flush=True)
# ---------------- structure + (a) + (b) + full copy check + lake/river bookkeeping, group by group
newb = sorted(p.name for p in N.iterdir() if (p/"zarr.json").is_file())
if newb != basins: P(f"group set differs: new {len(newb)} vs template {len(basins)}")
a_missing, a_extra_cols, a_dups = [], [], 0; b_nonfinite = {v: 0 for v in VARS}; b_nodes = set()
copy_all_ok, copy_all_n, copy_all_bad = 0, 0, []
meta_bad = []; ns_bad = []; cons_bad = []
river = dict(tpl_nodes=set(), new_nodes=set(), tpl_nodes_graph22=set(), new_nodes_graph=set()); rb = {}
RA = pd.read_csv(W/"river/reach_area_v35_all_reach_nodes.csv", dtype={"subbasin": str}); pairs = {r.subbasin: r.source.split(":")[1] for r in RA.itertuples() if r.source.startswith("v1_row_of_gone")}
pair_res = []
lake_cols = {}
LK = pd.read_parquet(CSF/"eval/v21_retrain/store/lake_targets_vd2026_halfexact.parquet"); lakes = sorted(set(LK.subbasin.astype(str))); lset = set(lakes)
v22 = {str(n["id"]) for n in json.load(open(C/"metadata/graphs_conus_v22.json"))["nodes"]}
def arr_meta(p):
    """Read the metadata file p of one Zarr array in a form that can be compared between stores.

    The shape and the sub-basin length of the chunks are removed, because they differ between
    the stores when the number of sub-basins differs. A data type given as a dictionary is
    reduced to its name.
    """
    j = json.load(open(p)); j.pop("shape"); cs = j["chunk_grid"]["configuration"]["chunk_shape"]; j["chunk_grid"]["configuration"]["chunk_shape"] = cs[:1] if len(cs) == 2 else []
    if isinstance(j["data_type"], dict): j["data_type"] = j["data_type"]["name"]
    return j
for b in basins:
    if not (N/b/"zarr.json").is_file(): P(f"{b}: group missing"); continue
    n = xr.open_zarr(N/b, consolidated=True); t = xr.open_zarr(T/b, consolidated=True)
    if not np.array_equal(n.date.values, t.date.values): P(f"{b}: dates differ")
    if sorted(n.data_vars) != sorted(t.data_vars): P(f"{b}: variables differ {sorted(n.data_vars)}")
    for v in list(n.data_vars) + ["date", "subbasin"]:
        if arr_meta(N/b/v/"zarr.json") != arr_meta(T/b/v/"zarr.json"): meta_bad.append(f"{b}/{v}")
    for v in n.data_vars:
        if json.load(open(N/b/v/"zarr.json"))["chunk_grid"]["configuration"]["chunk_shape"] != [365, n.sizes["subbasin"]]: meta_bad.append(f"{b}/{v} chunks")
    if json.dumps(dict(zarr.open_group(N/b, mode="r").attrs), sort_keys=True) != json.dumps(dict(zarr.open_group(T/b, mode="r").attrs), sort_keys=True): ns_bad.append(b)
    if not json.load(open(N/b/"zarr.json")).get("consolidated_metadata"): cons_bad.append(b)
    cols = [str(s) for s in n.subbasin.values]; cset = set(cols); a_dups += len(cols) - len(cset)
    rows = NM.index[NM.outlet_id == b].tolist()
    if cols != rows: a_extra_cols.append(b)
    gb = [s for s in rows if s in gset]; a_missing += [s for s in gb if s not in cset]
    gi = np.array([cols.index(s) for s in gb if s in cset], dtype=int) if gb else np.array([], int)
    tcols = [str(s) for s in t.subbasin.values]; tpos = {s: j for j, s in enumerate(tcols)}; npos = {s: j for j, s in enumerate(cols)}
    cp = [s for s in cols if NM.action[s] == "copy"]
    X = {v: n[v].transpose("date", "subbasin").values for v in VARS + TG}; Y = {v: t[v].transpose("date", "subbasin").values for v in VARS + TG}
    for v in VARS:
        bad = ~np.isfinite(X[v]); b_nonfinite[v] += int(bad.sum())
        if bad.any(): b_nodes |= {f"{b}:{cols[j]}" for j in np.flatnonzero(bad.any(0))}
    for s in cp:
        copy_all_n += 1; ok = all(X[v][:, npos[s]].tobytes() == Y[v][:, tpos[s]].tobytes() for v in VARS + ["dv_SWOTAE", "dv_staticPLD"])
        copy_all_ok += ok
        if not ok: copy_all_bad.append(f"{b}:{s}")
    for s in cols:
        if s in lset:
            lake_cols[s] = {v: (s in tpos) and X[v][:, npos[s]].tobytes() == Y[v][:, tpos[s]].tobytes() for v in ("dv_SWOTAE", "dv_staticPLD")}
            lake_cols[s]["template_has_data"] = (s in tpos) and bool(np.isfinite(Y["dv_SWOTAE"][:, tpos[s]]).any() or np.isfinite(Y["dv_staticPLD"][:, tpos[s]]).any())
    tr = np.isfinite(Y["dv_river"]).any(0); nr = np.isfinite(X["dv_river"]).any(0)
    rb[b] = dict(template_obs=int(np.isfinite(Y["dv_river"]).sum()), v35_obs=int(np.isfinite(X["dv_river"]).sum()), template_nodes=int(tr.sum()), v35_nodes=int(nr.sum()))
    for s, gsrc in pairs.items():
        if s in npos and gsrc in tpos:
            x, y = X["dv_river"][:, npos[s]], Y["dv_river"][:, tpos[gsrc]]; fx, fy = np.isfinite(x), np.isfinite(y)
            pair_res.append(dict(new=s, gone=gsrc, basin=b, days_template=int(fy.sum()), days_new=int(fx.sum()), same_days=bool((fx == fy).all()),
                                 days_equal=int(((x == y) & fx & fy).sum()), max_abs_diff=float(np.nanmax(np.abs(x - y))) if (fx & fy).any() else np.nan))
    river["tpl_nodes"] |= {tcols[j] for j in np.flatnonzero(tr)}; river["new_nodes"] |= {cols[j] for j in np.flatnonzero(nr)}
    R.setdefault("tot", {});
    for v in TG:
        R["tot"][f"{v}_finite_new"] = R["tot"].get(f"{v}_finite_new", 0) + int(np.isfinite(X[v]).sum())
        R["tot"][f"{v}_finite_template"] = R["tot"].get(f"{v}_finite_template", 0) + int(np.isfinite(Y[v]).sum())
    early = [v for v in TG if np.isfinite(n[v].sel(date=slice(None, "2022-12-31")).values).any()]
    if early: P(f"{b}: finite target before 2023 in {early}")
    print(b, len(cols), "cols", flush=True)
R["structure"] = dict(groups_new=len(newb), groups_template=len(basins), array_meta_mismatch=meta_bad[:20], n_array_meta_mismatch=len(meta_bad),
                      normalization_stats_differ=ns_bad, groups_not_consolidated=cons_bad, groups_cols_not_v35_rows=a_extra_cols)
R["a"] = dict(graph_nodes=len(gnodes), missing=len(a_missing), missing_first=a_missing[:20], duplicate_columns=a_dups,
              PASS=not a_missing and not a_dups and not a_extra_cols)
R["b"] = dict(nonfinite_values=b_nonfinite, nodes_with_nonfinite=sorted(b_nodes)[:30], n_nodes_with_nonfinite=len(b_nodes),
              template_nonfinite=json.load(open(W/"template_nan_scan.json"))["t2m_mean"]["nan_values"], PASS=sum(b_nonfinite.values()) == 0)
# ---------------- (c) 200 random copied GRAPH nodes, all 13 variables, bitwise
cg = NM[(NM.action == "copy") & NM.index.isin(gset)]; pick = rng.choice(cg.index.values, 200, replace=False); c_bad = []
for s in pick:
    b = cg.outlet_id[s]; n = xr.open_zarr(N/b, consolidated=True); t = xr.open_zarr(T/b, consolidated=True)
    for v in VARS + TG:
        x = n[v].sel(subbasin=s).values; y = t[v].sel(subbasin=s).values
        if x.dtype != y.dtype or x.tobytes() != y.tobytes(): c_bad.append(f"{b}:{s}:{v}")
R["c"] = dict(sampled=200, seed=35, variables=len(VARS + TG), mismatches=c_bad[:20], n_mismatch=len(c_bad), all_copied_nodes=copy_all_n,
              all_copied_nodes_bitwise_equal_era5_and_lake=copy_all_ok, all_copied_bad=copy_all_bad[:20], PASS=not c_bad and copy_all_ok == copy_all_n)
# ---------------- (d) re-extracted nodes sharing an id with a template node
sid = NM[(NM.action == "extract") & NM.in_template.astype(str).eq("True")]
rows = []
for s in sid.index:
    b = sid.outlet_id[s]
    rows.append((b, s))
D = []
byb = {}
for b, s in rows: byb.setdefault(b, []).append(s)
for b, ss in byb.items():
    n = xr.open_zarr(N/b, consolidated=True); t = xr.open_zarr(T/b, consolidated=True)
    for v in ("t2m_mean", "tp_mean"):
        x = n[v].sel(subbasin=ss).transpose("date", "subbasin").values; y = t[v].sel(subbasin=ss).transpose("date", "subbasin").values
        for j, s in enumerate(ss):
            r = float(np.corrcoef(x[:, j], y[:, j])[0, 1])
            D.append(dict(subbasin=s, basin=b, var=v, r=r, mean_new=float(x[:, j].mean()), mean_tpl=float(y[:, j].mean()), in_graph=s in gset,
                          area_ratio=float(NM.area_ratio_vs_template[s]) if NM.area_ratio_vs_template[s] != "" else np.nan))
D = pd.DataFrame(D); D.to_csv(W/"checks_d_all_same_id_extract.csv", index=False)
def plaus(g):
    """Return whether the mean temperature and precipitation of one node are plausible.

    g holds the rows of one node in the table D (one row per variable). The bounds are: mean t2m
    in [250, 305] K and within 3 K of the template; mean tp in (0, 0.02] m/day and between 0.5
    and 2 times the template. The same bounds are written to the report of gate (d).
    """
    t2, tp = g[g["var"] == "t2m_mean"].iloc[0], g[g["var"] == "tp_mean"].iloc[0]
    return (250 <= t2.mean_new <= 305) and abs(t2.mean_new - t2.mean_tpl) <= 3.0 and (0 < tp.mean_new <= 0.02) and (0.5 <= tp.mean_new/tp.mean_tpl <= 2.0)
Dg = D[D.in_graph]; ids = Dg.subbasin.unique(); pick_d = rng.choice(ids, min(100, len(ids)), replace=False)
res = []
for s in sorted(set(D.subbasin)):
    g = D[D.subbasin == s]; r_t2, r_tp = float(g[g["var"] == "t2m_mean"].r.iloc[0]), float(g[g["var"] == "tp_mean"].r.iloc[0])
    res.append(dict(subbasin=s, r_t2m=r_t2, r_tp=r_tp, plausible=plaus(g), ok=(r_t2 > 0.95) and (r_tp > 0.95) and plaus(g), sampled=s in set(pick_d),
                    area_ratio=float(g.area_ratio.iloc[0]), in_graph=bool(g.in_graph.iloc[0])))
res = pd.DataFrame(res); smp = res[res.sampled]
R["d"] = dict(candidates_same_id_reextracted=len(res), candidates_in_graph=int(res.in_graph.sum()), sampled=len(smp), seed=35,
              sample_pass=int(smp.ok.sum()), sample_min_r_t2m=float(smp.r_t2m.min()), sample_min_r_tp=float(smp.r_tp.min()),
              sample_fail=smp[~smp.ok][["subbasin", "r_t2m", "r_tp", "plausible", "area_ratio"]].to_dict("records"),
              all_pass=int(res.ok.sum()), all_fail=res[~res.ok].sort_values("r_tp")[["subbasin", "r_t2m", "r_tp", "plausible", "area_ratio", "in_graph"]].head(25).to_dict("records"),
              all_n_fail=int((~res.ok).sum()), all_median_r_tp=float(res.r_tp.median()), all_median_r_t2m=float(res.r_t2m.median()),
              bounds="t2m mean in [250,305] K and |mean - template| <= 3 K; tp mean in (0, 0.02] m/day and new/template in [0.5, 2]",
              PASS=bool(smp.ok.all()))
# ---------------- (d2) method check at copied nodes: extraction vs template column
SB = pd.read_csv(W/"subbasins.csv", dtype=str); pos = {s: i for i, s in enumerate(SB.subbasin)}
susq = set(json.load(open(CSF/"eval/overnight/era5_2026/susq_map.json")))
yrs = [(y, np.load(W/f"years/{y}.npy")) for y in range(2020, 2026)] + [(2026, np.load(W/"years/2026_raw.npy"))]
worst = {"2020-25": {}, "2026": {}}; exact26 = {}; wnode = {}; nbig = {}; exceed = {}
KNOWN = {"7310084002": "2-cell reservoir in USGS-01578310 (Susquehanna): the PRODUCTION extraction (template 2020-25) differs from the era5ext method in 2020/2022/2023 only (<= 0.42 sd in sro; 2021/2024/2025 agree to 1e-6 sd); pre-existing in the template; copied bitwise like every identical-polygon node"}
for b in basins:
    cols = [s for s in NM.index[NM.outlet_id == b] if NM.action[s] == "copy" and s not in susq]
    if not cols: continue
    t = xr.open_zarr(T/b, consolidated=True); ii = np.array([pos[s] for s in cols])
    for k, v in enumerate(VARS):
        y_ = t[v].sel(subbasin=cols).transpose("date", "subbasin").values
        x_ = np.concatenate([A[:, ii, k] for _, A in yrs]); i26 = 2192
        sd = float(np.std(y_)) or 1.0
        pn = np.max(np.abs(x_ - y_), axis=0)/sd; nbig[v] = nbig.get(v, 0) + int((pn > 1e-3).sum())
        for j in np.flatnonzero(pn > 1e-4): exceed.setdefault(cols[j], {})[v] = float(pn[j])
        keep = np.array([c not in KNOWN for c in cols])
        if not keep.any(): continue
        d1 = float(np.max(np.abs(x_[:i26, keep] - y_[:i26, keep])))/sd; d2 = float(np.max(np.abs(x_[i26:] - y_[i26:])))/sd
        if d1 > worst["2020-25"].get(v, 0): wnode[v] = f"{b}:{np.array(cols)[keep][int(np.argmax(pn[keep]))]}"
        worst["2020-25"][v] = max(worst["2020-25"].get(v, 0), d1); worst["2026"][v] = max(worst["2026"].get(v, 0), d2)
        exact26[v] = exact26.get(v, True) and bool(np.array_equal(x_[i26:], y_[i26:]))
R["d2"] = dict(note="our v3.5-polygon extraction at copied nodes (Susquehanna 2 excluded: template maps them via susq_map.json) vs template; max|diff|/sd",
               worst_2020_25_excl_known=worst["2020-25"], worst_node_2020_25=wnode, nodes_exceeding_1e4_sd={k: max(v.values()) for k, v in exceed.items()}, known_exceptions=KNOWN, nodes_over_1e3_sd=nbig, worst_2026=worst["2026"], bitwise_2026=exact26,
               PASS=set(exceed) <= set(KNOWN) and max(worst["2026"].values()) == 0)
# ---------------- (d3) weight rows
Z1 = np.load(EXT/"weights.npz"); Z2 = np.load(W/"weights.npz"); S1 = pd.read_csv(EXT/"subbasins.csv", dtype=str); p1 = {s: i for i, s in enumerate(S1.subbasin)}
def row(Z, i):
    """Return the grid-cell indices and the weights of row i of the weight file Z (CSR form)."""
    a, e = Z["ptr"][i], Z["ptr"][i + 1]; return Z["cell"][a:e], Z["w"][a:e]
cp_all = NM.index[NM.action == "copy"]; wbad = [s for s in cp_all if not all(np.array_equal(u, v) for u, v in zip(row(Z1, p1[s]), row(Z2, pos[s])))]
ex_same = NM.index[(NM.action == "extract") & NM.in_template.astype(str).eq("True")]
wsame_ex = [s for s in ex_same if all(np.array_equal(u, v) for u, v in zip(row(Z1, p1[s]), row(Z2, pos[s])))]
R["d3"] = dict(copied_nodes=len(cp_all), weight_rows_differ=len(wbad), first=wbad[:10], reextracted_same_id_with_identical_weight_row=len(wsame_ex),
               PASS=not wbad)
# ---------------- (e) lakes
e_bad = [s for s in lakes if not (s in lake_cols and lake_cols[s]["dv_SWOTAE"] and lake_cols[s]["dv_staticPLD"])]
tot = R["tot"]
R["e"] = dict(lakes=len(lakes), lakes_found=int(sum(s in lake_cols for s in lakes)), lakes_template_has_data=int(sum(lake_cols.get(s, {}).get("template_has_data", False) for s in lakes)), lakes_not_bitwise=e_bad[:20], n_bad=len(e_bad),
              dv_SWOTAE_finite=(tot["dv_SWOTAE_finite_new"], tot["dv_SWOTAE_finite_template"]), dv_staticPLD_finite=(tot["dv_staticPLD_finite_new"], tot["dv_staticPLD_finite_template"]),
              lakes_in_v35_graph=int(sum(s in gset for s in lakes)), lakes_outside_graph_rows=[s for s in lakes if s not in gset][:20],
              PASS=not e_bad and tot["dv_SWOTAE_finite_new"] == tot["dv_SWOTAE_finite_template"] and tot["dv_staticPLD_finite_new"] == tot["dv_staticPLD_finite_template"])
# ---------------- dv_river
tn, nn = river["tpl_nodes"], river["new_nodes"]
lost = sorted(tn - nn); gained = sorted(nn - tn)
lost_gone = [s for s in lost if s not in NM.index]; lost_present = [s for s in lost if s in NM.index]
RB = pd.DataFrame(rb).T; RB["lost_obs"] = RB.template_obs - RB.v35_obs; RB["loss_pct"] = 100*RB.lost_obs/RB.template_obs.where(RB.template_obs > 0)
RB.index.name = "basin"; RB.to_csv(W/"checks_river_by_basin.csv"); PR = pd.DataFrame(pair_res); PR.to_csv(W/"checks_river_new_vs_gone.csv", index=False)
R["river_obs"] = dict(template_obs=int(RB.template_obs.sum()), v35_obs=int(RB.v35_obs.sum()), net_lost_obs=int(RB.lost_obs.sum()),
                      basins_loss_over_1pct=RB[RB.loss_pct > 1].sort_values("loss_pct", ascending=False)[["template_obs", "v35_obs", "lost_obs", "loss_pct", "template_nodes", "v35_nodes"]].round(2).reset_index().to_dict("records"),
                      basins_gain=int((RB.lost_obs < 0).sum()),
                      new_id_vs_gone_v1_node_pairs=len(PR), pairs_same_obs_days=int(PR.same_days.sum()) if len(PR) else 0,
                      pairs_all_days_bitwise=int((PR.days_equal == PR.days_template).sum()) if len(PR) else 0,
                      pairs_obs_template=int(PR.days_template.sum()) if len(PR) else 0, pairs_obs_equal=int(PR.days_equal.sum()) if len(PR) else 0)
R["river"] = dict(template_nodes_with_dv_river=len(tn), v35_nodes_with_dv_river=len(nn), common=len(tn & nn), lost=len(lost), lost_id_gone_from_v35=len(lost_gone),
                  lost_id_present=lost_present[:20], gained=len(gained), gained_first=gained[:20],
                  template_graph22_nodes_with_dv_river=len({s for s in tn if s in v22}), v35_graph_nodes_with_dv_river=len({s for s in nn if s in gset}),
                  lost_graph22_to_graph35=len({s for s in tn if s in v22} - {s for s in nn if s in gset}),
                  gained_graph=len({s for s in nn if s in gset} - {s for s in tn if s in v22}),
                  finite_values=(tot["dv_river_finite_new"], tot["dv_river_finite_template"]))
# ---------------- (f)
du = subprocess.run(["du", "-sb", str(N)], capture_output=True, text=True).stdout.split()[0]
dh = subprocess.run(["du", "-sh", str(N)], capture_output=True, text=True).stdout.split()[0]
tdu = subprocess.run(["du", "-sh", str(T)], capture_output=True, text=True).stdout.split()[0]
fs = shutil.disk_usage(f"{WORK_FS}"); wk = subprocess.run(["du", "-sh", str(W)], capture_output=True, text=True).stdout.split()[0]
R["f"] = dict(store_bytes=int(du), store_h=dh, template_h=tdu, work_dir_h=wk, work_free_GB=round(fs.free/1e9, 1), PASS=fs.free/1e9 >= 100)
R["problems"] = prob
R["ALL_PASS"] = all(R[k]["PASS"] for k in ("a", "b", "c", "d", "d2", "d3", "e", "f")) and not prob and not meta_bad and not ns_bad and not cons_bad and not a_extra_cols and newb == basins
json.dump(R, open(F/"checks.json", "w"), indent=1, default=str)
print(json.dumps(R, indent=1, default=str))
