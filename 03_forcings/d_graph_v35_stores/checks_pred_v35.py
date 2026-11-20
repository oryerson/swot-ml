"""Acceptance checks ("gates") for the two graph v3.5 prediction stores.

Each store is compared with its template, the store of the same period on the earlier node set.
All stores are only read. The gates are those of the training store (checks_v35.py), plus
cross-store checks. `ALL_PASS` of a store is true only if all its gates and the overlap check pass.

Reads (paths relative to $SWOTGAT_WORK; <forcing> = CONUS_storage_fork/eval/v35_inputs/forcing)
    <forcing>/zarr_store_ext1980_v35   against the template CONUS/zarr_store_ext1980 (1980-2025)
    <forcing>/zarr_store_vd2026_v35    against the template CONUS/zarr_store_vd2026 (2020-2026)
    <forcing>/zarr_store_vd2026_halfexact_v35     the graph v3.5 training store
    <forcing>/work/node_map.csv                   copy or extract per sub-basin (node_map.py)
    <forcing>/work_pred/subbasins.csv, work_pred/years/1980.npy .. 2019.npy   the 1980-2019
                                                  extraction (extract_pred_v35.py)
    CONUS_storage_fork/eval/graph_v3/v35/graphs_conus_v35.json    graph v3.5 (its node list)

Writes
    <forcing>/checks_pred.json                    all results (also printed)
    <forcing>/work_pred/checks_d_<store>.csv      gate (d), the sampled nodes
    <forcing>/MANIFEST_<store>_v35_group_zarr_json.md5   md5 of every group's zarr.json

Usage: python checks_pred_v35.py     (no arguments; run after all build_store_pred_v35.py jobs)

Gates, per store
    structure  102 groups in store and template. Same date axis, same variables per group, same
               array metadata (data type, codecs, fill value, 365-day chunks), identical group
               attributes, and the same consolidation state as the template.
    (a)   Every graph v3.5 node is a column of its basin group. The columns are exactly the v3.5
          sub-basin rows of the basin, in the order of the sub-basin table, with no duplicates,
          and in the same order as in the training store.
    (b)   All 10 ERA5-Land means are finite at every column and date. (The count of non-finite
          values in the template is reported.)
    (c)   Every copied column is bitwise equal to the template in every variable. The comparison
          is on the int64 view of the float64 values, so the bit pattern of a NaN counts too.
          A sample of 200 random graph nodes is also reported.
    (d)   100 random graph nodes that were extracted again but share an id with a template column:
          over the full date range, daily t2m and tp correlate with the template above 0.95, and
          the means are plausible (t2m mean within 250-305 K and within 3 K of the template; tp
          mean in (0, 0.02] m/day and between 0.5 and 2 times the template).
    (d2)  ext1980 only. Method check on the 1,000 copied validation nodes of make_pred_weights.py:
          the new 1980-2019 extraction against the template column. For 1980-1999 the template
          was made with the same method, so the values must be bitwise equal. For 2000-2019 the
          template comes from the production extraction; no node may differ by more than 1e-4
          standard deviations (max |difference| / standard deviation of the template values of
          the basin; the typical difference is about 1e-6). Nodes listed in KNOWN are left out.
    (e)   Every other variable except dv_river (the lake targets and water levels) is bitwise
          equal to the template at every retained id that has template data.
    (x)   vd2026 only: the ERA5-Land means and dv_river are bitwise equal to the training store
          (the two templates have the same relation).
    river Reported, not a gate: nodes and observations of dv_river against the template, new
          nodes with data, and basins that lose more than 1 % of their observations.
    (f)   Store sizes are reported; the work file system (WORK_FS in swotgat.paths) must have at
          least 100 GB free.

Check across the two stores
    overlap  The ERA5-Land means of ext1980_v35 and vd2026_v35 are bitwise equal on their common
             period, 2020-01-01 to 2025-12-31. The same test on the two templates is reported.
             The long-memory inputs (e_long_memory) join the two stores and rely on this.

Notes
    KNOWN documents two nodes of the Susquehanna basin USGS-01578310 (7310084002 and 73005469)
    where the production extraction differs from the method used here in some years. Production
    is consistent with itself there (the two templates agree on 2020-2025), and both nodes are
    copied bit for bit like every node with an unchanged polygon.
"""
from swotgat.paths import WORK, WORK_FS
import json, hashlib, shutil, subprocess, numpy as np, pandas as pd, xarray as xr, zarr
from pathlib import Path
ROOT = Path(f"{WORK}"); C = ROOT/"CONUS"; CSF = ROOT/"CONUS_storage_fork"
F = CSF/"eval/v35_inputs/forcing"; W = F/"work"; P = F/"work_pred"; V35 = CSF/"eval/graph_v3/v35"; HX = F/"zarr_store_vd2026_halfexact_v35"
VARS = [f"{v}_mean" for v in ["ssrd", "strd", "ro", "sro", "e", "tp", "sp", "d2m", "t2m", "fal"]]
ST = {"ext1980": (C/"zarr_store_ext1980", F/"zarr_store_ext1980_v35", False, ("1980-01-01", "2025-12-31")),
      "vd2026": (C/"zarr_store_vd2026", F/"zarr_store_vd2026_v35", True, ("2020-01-01", "2026-09-30"))}
NM = pd.read_csv(W/"node_map.csv", dtype={"subbasin": str, "outlet_id": str, "template_group": str}, keep_default_na=False).set_index("subbasin")
gset = {str(n["id"]) for n in json.load(open(V35/"graphs_conus_v35.json"))["nodes"]}
KNOWN = {"7310084002": "2-cell reservoir in USGS-01578310 (Susquehanna): production 2020/22/23 differs from the era5ext method (<= 0.42 sd in sro); copied bitwise",
         "73005469": "split_downstream node (36.9 km2) in USGS-01578310 (Susquehanna): production 2000-2019 differs from the era5ext method (<= 5.5 sd in sro); production is self-consistent (ext1980 == vd2026 for 2020-25); copied bitwise (hub / checker B F2, 2026-10-22)"}
def arr_meta(p):
    """Read the metadata file p of one Zarr array in a form that can be compared between stores.

    The shape and the sub-basin length of the chunks are removed, because they differ between
    the stores when the number of sub-basins differs. A data type given as a dictionary is
    reduced to its name.
    """
    j = json.load(open(p)); j.pop("shape"); cs = j["chunk_grid"]["configuration"]["chunk_shape"]; j["chunk_grid"]["configuration"]["chunk_shape"] = cs[:1] if len(cs) == 2 else []
    if isinstance(j["data_type"], dict): j["data_type"] = j["data_type"]["name"]
    return j
def iv(a):
    """Return the int64 view of a float64 array, for bitwise comparison.

    Two NaN values compare equal only if they have the same bits.
    """
    return np.ascontiguousarray(a).view(np.int64)
OUT = {}; rng = np.random.default_rng(2026)
for name, (T, N, CONS, RANGE) in ST.items():
    R = dict(structure=dict(meta_mismatch=[], attrs_differ=[], consolidation_differs=[], varset_differs=[], dates_differ=[]),
             a=dict(missing=[], dup=0, cols_not_rows=[], order_differs_from_training_store=[]), b={v: 0 for v in VARS}, b_template={v: 0 for v in VARS},
             c=dict(cols=0, bad=[]), e=dict(ids=0, bad=[]), x=dict(), river=dict(template_nodes=0, v35_nodes=0, template_obs=0, v35_obs=0, new_id_nodes_with_data=0, lost_by_basin={}))
    copybit = {}; xbad = []
    basins = sorted(p.name for p in T.iterdir() if (p/"zarr.json").is_file())
    nb = sorted(p.name for p in N.iterdir() if (p/"zarr.json").is_file()); R["structure"]["groups"] = [len(nb), len(basins)]
    for b in basins:
        n = xr.open_zarr(N/b, consolidated=CONS); t = xr.open_zarr(T/b, consolidated=CONS)
        if not np.array_equal(n.date.values, t.date.values) or not pd.DatetimeIndex(n.date.values).equals(pd.date_range(*RANGE)): R["structure"]["dates_differ"].append(b)
        if sorted(n.data_vars) != sorted(t.data_vars): R["structure"]["varset_differs"].append(b)
        for v in list(n.data_vars) + ["date", "subbasin"]:
            if arr_meta(N/b/v/"zarr.json") != arr_meta(T/b/v/"zarr.json"): R["structure"]["meta_mismatch"].append(f"{b}/{v}")
        if json.dumps(dict(zarr.open_group(N/b, mode="r").attrs), sort_keys=True) != json.dumps(dict(zarr.open_group(T/b, mode="r").attrs), sort_keys=True): R["structure"]["attrs_differ"].append(b)
        if bool(json.load(open(N/b/"zarr.json")).get("consolidated_metadata")) != bool(json.load(open(T/b/"zarr.json")).get("consolidated_metadata")): R["structure"]["consolidation_differs"].append(b)
        cols = [str(s) for s in n.subbasin.values]; rows = NM.index[NM.outlet_id == b].tolist(); R["a"]["dup"] += len(cols) - len(set(cols))
        if cols != rows: R["a"]["cols_not_rows"].append(b)
        if cols != [str(s) for s in xr.open_zarr(HX/b, consolidated=True).subbasin.values]: R["a"]["order_differs_from_training_store"].append(b)
        R["a"]["missing"] += [s for s in rows if s in gset and s not in set(cols)]
        tcols = [str(s) for s in t.subbasin.values]; tpos = {s: j for j, s in enumerate(tcols)}
        cp = [j for j, s in enumerate(cols) if NM.action[s] == "copy"]; cpt = [tpos[cols[j]] for j in cp]
        sid = [j for j, s in enumerate(cols) if s in tpos]; sidt = [tpos[cols[j]] for j in sid]
        for v in n.data_vars:
            X = n[v].transpose("date", "subbasin").values; Y = t[v].transpose("date", "subbasin").values
            if v in VARS:
                R["b"][v] += int((~np.isfinite(X)).sum()); R["b_template"][v] += int((~np.isfinite(Y)).sum())
                if cp:
                    eq = (iv(X[:, cp]) == iv(Y[:, cpt])).all(0)
                    for j, e in zip(cp, eq): copybit.setdefault(cols[j], {})[v] = bool(e)
                if name == "vd2026":
                    H = xr.open_zarr(HX/b, consolidated=True)[v].transpose("date", "subbasin").values
                    if not (iv(X) == iv(H)).all(): xbad.append(f"{b}/{v} != training store")
            else:
                if sid:
                    eq = (iv(X[:, sid]) == iv(Y[:, sidt])).all(0)
                    for j, e in zip(sid, eq):
                        if NM.action[cols[j]] == "copy": copybit.setdefault(cols[j], {})[v] = bool(e)
                        if v != "dv_river" and np.isfinite(Y[:, tpos[cols[j]]]).any():
                            R["e"]["ids"] += 1
                            if not e: R["e"]["bad"].append(f"{b}:{cols[j]}:{v}")
                if v == "dv_river":
                    tr, nr = np.isfinite(Y), np.isfinite(X)
                    R["river"]["template_nodes"] += int(tr.any(0).sum()); R["river"]["v35_nodes"] += int(nr.any(0).sum())
                    R["river"]["template_obs"] += int(tr.sum()); R["river"]["v35_obs"] += int(nr.sum())
                    R["river"]["new_id_nodes_with_data"] += int(sum(nr[:, j].any() for j, s in enumerate(cols) if s not in tpos))
                    if tr.sum() and (tr.sum() - nr.sum())/tr.sum() > 0.01: R["river"]["lost_by_basin"][b] = [int(tr.sum()), int(nr.sum()), round(100*(tr.sum() - nr.sum())/tr.sum(), 2)]
                    if name == "vd2026":
                        H = xr.open_zarr(HX/b, consolidated=True)["dv_river"].transpose("date", "subbasin").values
                        if not (iv(X) == iv(H)).all(): xbad.append(f"{b}/dv_river != training store")
            del X, Y
        print(name, b, len(cols), flush=True)
    R["x"]["bad"] = xbad
    cbad = [s for s, d in copybit.items() if not all(d.values())]
    cg = [s for s in copybit if s in gset]; pick = rng.choice(cg, 200, replace=False)
    R["c"] = dict(all_copied_cols=len(copybit), all_bitwise=len(copybit) - len(cbad), bad=cbad[:20], sample=200, sample_vars_per_node=int(np.median([len(copybit[s]) for s in pick])),
                  sample_bad=[s for s in pick if not all(copybit[s].values())], PASS=not cbad)
    R["a"]["PASS"] = not R["a"]["missing"] and not R["a"]["dup"] and not R["a"]["cols_not_rows"] and not R["a"]["order_differs_from_training_store"]
    R["a"]["missing"] = R["a"]["missing"][:20]
    R["b"] = dict(nonfinite=R["b"], template_nonfinite=R["b_template"], PASS=sum(R["b"].values()) == 0); R.pop("b_template")
    R["e"]["PASS"] = not R["e"]["bad"]; R["e"]["bad"] = R["e"]["bad"][:20]
    # (d) correlation sample
    sid_ex = NM[(NM.action == "extract") & NM.in_template.astype(str).eq("True") & NM.index.isin(gset)]
    pick_d = rng.choice(sid_ex.index.values, 100, replace=False); D = []
    for s in pick_d:
        b = sid_ex.outlet_id[s]; n = xr.open_zarr(N/b, consolidated=CONS); t = xr.open_zarr(T/b, consolidated=CONS); d = dict(subbasin=s)
        for v in ("t2m_mean", "tp_mean"):
            x = n[v].sel(subbasin=s).values; y = t[v].sel(subbasin=s).values; d[f"r_{v[:-5]}"] = float(np.corrcoef(x, y)[0, 1]); d[f"m_{v[:-5]}"] = (float(x.mean()), float(y.mean()))
        d["ok"] = (d["r_t2m"] > 0.95 and d["r_tp"] > 0.95 and 250 <= d["m_t2m"][0] <= 305 and abs(d["m_t2m"][0] - d["m_t2m"][1]) <= 3 and 0 < d["m_tp"][0] <= 0.02 and 0.5 <= d["m_tp"][0]/d["m_tp"][1] <= 2)
        D.append(d)
    D = pd.DataFrame(D); D.to_csv(F/f"work_pred/checks_d_{name}.csv", index=False)
    R["d"] = dict(sampled=100, candidates=len(sid_ex), passed=int(D.ok.sum()), min_r_t2m=float(D.r_t2m.min()), min_r_tp=float(D.r_tp.min()),
                  fail=D[~D.ok][["subbasin", "r_t2m", "r_tp"]].to_dict("records"), PASS=bool(D.ok.all()))
    # (d2) ext1980: validation sample vs template, 1980-2019
    if name == "ext1980":
        S1 = pd.read_csv(P/"subbasins.csv", dtype=str); val = S1[S1.role == "validation_copy"]; pos1 = {s: i for i, s in enumerate(S1.subbasin)}
        worst = {"1980-99": {}, "2000-19": {}}; bitw = {}; exceed = {}
        for b, g in val.groupby("outlet_id"):
            cols = [s for s in g.subbasin if s not in KNOWN]
            if not cols: continue
            t = xr.open_zarr(T/b, consolidated=False); ii = np.array([pos1[s] for s in cols])
            for k, v in enumerate(VARS):
                y_ = t[v].sel(subbasin=cols, date=slice("1980-01-01", "2019-12-31")).transpose("date", "subbasin").values
                x_ = np.concatenate([np.load(P/f"years/{y}.npy", mmap_mode="r")[:, ii, k] for y in range(1980, 2020)]); sd = float(np.std(y_)) or 1.0
                i0 = 7305; d1 = float(np.max(np.abs(x_[:i0] - y_[:i0])))/sd; d2 = float(np.max(np.abs(x_[i0:] - y_[i0:])))/sd
                worst["1980-99"][v] = max(worst["1980-99"].get(v, 0), d1); worst["2000-19"][v] = max(worst["2000-19"].get(v, 0), d2)
                bitw[v] = bitw.get(v, True) and bool((iv(x_[:i0]) == iv(y_[:i0])).all())
                pn = np.max(np.abs(x_ - y_), axis=0)/sd
                for j in np.flatnonzero(pn > 1e-4): exceed.setdefault(cols[j], {})[v] = float(pn[j])
        R["d2"] = dict(validation_nodes=len(val), excluded_known=sorted(set(KNOWN) & set(val.subbasin)), known_exceptions={k: KNOWN[k] for k in sorted(set(KNOWN) & set(val.subbasin))}, worst_max_abs_diff_over_sd=worst, bitwise_1980_99=bitw,
                       nodes_exceeding_1e4_sd={k: max(v.values()) for k, v in exceed.items()}, PASS=all(bitw.values()) and not exceed)
    # the overlap of ext1980_v35 and vd2026_v35 is checked once, below, after both stores
    OUT[name] = R
# overlap of the two stores, ERA5-Land 2020-2025 (for the templates and for the v3.5 stores)
ov = dict(v35_bad=[], template_bad=[])
for b in sorted(p.name for p in ST["ext1980"][1].iterdir() if (p/"zarr.json").is_file()):
    for tag, (s1, s2) in (("v35", (ST["ext1980"][1], ST["vd2026"][1])), ("template", (ST["ext1980"][0], ST["vd2026"][0]))):
        a = xr.open_zarr(s1/b, consolidated=False); c = xr.open_zarr(s2/b, consolidated=True)
        if list(a.subbasin.values.astype(str)) != list(c.subbasin.values.astype(str)): ov[f"{tag}_bad"].append(f"{b}: subbasin order"); continue
        for v in VARS:
            x = a[v].sel(date=slice("2020-01-01", "2025-12-31")).values; y = c[v].sel(date=slice("2020-01-01", "2025-12-31")).values
            if not (iv(x) == iv(y)).all(): ov[f"{tag}_bad"].append(f"{b}/{v}")
OUT["overlap_2020_2025"] = dict(v35_bad=ov["v35_bad"][:20], n_v35_bad=len(ov["v35_bad"]), template_bad=ov["template_bad"][:20], n_template_bad=len(ov["template_bad"]), PASS=not ov["v35_bad"])
# manifests of the group metadata, and gate (f)
for name, (T, N, CONS, RANGE) in ST.items():
    groups = sorted(p.name for p in N.iterdir() if (p/"zarr.json").is_file())
    man = [(f"{g}/zarr.json", hashlib.md5((N/g/"zarr.json").read_bytes()).hexdigest()) for g in groups]
    txt = "".join(f"{h}  {p}\n" for p, h in man); (F/f"MANIFEST_{name}_v35_group_zarr_json.md5").write_text(txt)
    du = subprocess.run(["du", "-sb", str(N)], capture_output=True, text=True).stdout.split()[0]
    OUT[name]["f"] = dict(store_bytes=int(du), template_h=subprocess.run(["du", "-sh", str(T)], capture_output=True, text=True).stdout.split()[0],
                          root_zarr_json=(N/"zarr.json").is_file(), manifest_md5=hashlib.md5(txt.encode()).hexdigest(), groups=len(groups))
    OUT[name]["x"]["PASS"] = not OUT[name]["x"]["bad"]
    s = OUT[name]["structure"]; s["PASS"] = s["groups"][0] == s["groups"][1] == 102 and not any(s[k] for k in ("meta_mismatch", "attrs_differ", "consolidation_differs", "varset_differs", "dates_differ"))
fs = shutil.disk_usage(f"{WORK_FS}"); OUT["work_free_GB"] = round(fs.free/1e9, 1)
for name in ST:
    OUT[name]["f"]["PASS"] = OUT["work_free_GB"] >= 100
    OUT[name]["ALL_PASS"] = all(OUT[name][k]["PASS"] for k in OUT[name] if isinstance(OUT[name][k], dict) and "PASS" in OUT[name][k]) and OUT["overlap_2020_2025"]["PASS"]
json.dump(OUT, open(F/"checks_pred.json", "w"), indent=1, default=str); print(json.dumps(OUT, indent=1, default=str))
