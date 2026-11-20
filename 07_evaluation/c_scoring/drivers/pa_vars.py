"""Build the table of candidate drivers of skill: one row of descriptors per scored reservoir.

Each variable describes a reservoir's climate, dam, upstream regulation, basin, gauge behavior,
SWOT data quality or gauge quality. The table reads no model output and no score, so that skill
can be related to it afterwards. The list of variables was fixed before skill was looked at.

Usage: bin/run 07_evaluation/c_scoring/drivers/pa_vars.py      (no arguments; after pa_blocks.py)

Reads (through pa_lib.py; see the note there)
  The reservoir list comes from an earlier model configuration's score table (ids only; 482
  reservoirs), the network and static attributes from graph v2.2, an earlier graph version than
  the paper's v3.5, and the gauge test sets from the copy in eval/g4_fix/testsets ("the 1980-2020
  test set" and "the 2023-2026 test set" below). Further inputs are named in the groups: Global
  Dam Watch (GDW) v1.0, ResOpsUS inflow, outflow and attributes, the SWOT target store and the
  list of lake nodes that carry a SWOT target ("the SWOT node list"), the 1980-2025 forcing store,
  a long-memory feature store, out/blocks.csv (pa_blocks.py, for lat and lon) and
  eval/a180_graphdiff/classes.csv (a comparison of graph versions made for an earlier
  configuration).

Writes, to out/ next to pa_lib.py
  vars.parquet    one row per DAM_ID, one column per variable
  vars_dict.csv   per column: group, description, number of valid values
  reps.json       up to three representative variables per group (a representative with more than
                  60% missing values is replaced by its fallback, if it has one), notes on
                  coverage, and the list of variables that use gauge data of the scored window

Conventions
  Upstream = every graph v2.2 node that drains to the reservoir's node (column `node` of the
  1980-2020 test set's sites table), including that node. Variables ending in `_up` are
  area-weighted means over the upstream nodes; `_loc` values are those of the reservoir's own
  node. The fields are HydroATLAS attributes in the static table; values <= -999 are missing.
  A water year (WY) runs from October to September.

Groups:
 A climate      ai_up/ai_loc (ari_ix_sav/100, P/PET), cmi_up, cmi_min_up (driest-month CMI),
                snow_up/snowmax_up (snw_pc_syr/smx), pseas_up (Walsh-Lawler seasonality index of
                the upstream monthly P), ppet_lag_up (months between the upstream P peak and PET
                peak, 0-6), tmp_up (degC), pre_up, pet_up, aet_ratio_up (AET/P), run_up (mm/yr),
                gla_up
 B clim. shift  from the long-memory feature store features/longmem_vd2026 at the reservoir node
                (built from ERA5 on the first graph version's upstream areas):
                shift_spei12 / shift_ssi12 (mean over 2010-20 minus mean over 2023-03..2026-09),
                abs_shift_spei12, out_lk24 / out_lk6 (share of 2010-20 days with lm_up_lk24 /
                lm_up_lk6 outside the 2023-26 min..max), dry_frac / wet_frac (share of 2010-20
                days with lm_up_spei12 < -1.5 / > 1.5)
 C reservoir    GDW v1.0 by GRAND_ID = DAM_ID: log_cap, log_area, log_depth (CAP/AREA, m), depth_gdw,
                log_hgt, age (2015 - YEAR_DAM), log_dor_gdw (DOR_PC, %), log_vq_obs (capacity /
                ResOpsUS mean inflow, water years with >= 292 days, >= 5 of them), elev_dam,
                log_power, n_uses, use_irri/elec/supp/fcon/recr/navi (any rank), main_* (MAIN_USE
                one-hot), lake_ctrl, n_members, log_cap_per_uparea
 D regulation   graph v2.2 + the static column dam_cap_mcm: log1p_up_dams / log_up_cap (GDW-matched
                dam nodes upstream, own members excluded), up_dor_cum (upstream capacity / mean
                annual flow at the node, dis_m3_pyr), n_down_res (dam nodes downstream to the
                outlet), log1p_up_swot (lake nodes on the SWOT node list upstream, own members
                excluded), log_dor_ha (HydroATLAS dor_pc_pva), graph_changed / graph_fixed /
                area_err_v22 (from eval/a180_graphdiff/classes.csv: graph v1 to v2.2 change class
                G-minor or G-major; wrong in v1 and correct in v2.2; |error| of the v2.2 upstream
                area against MERIT), catch_lratio (|log10 GDW CATCH_SKM / graph upstream area|)
 E basin        log_uparea, ele_up, ele_loc, relief_loc, slp_up, sgr_loc, cly/slt/snd/soc/kar/gwt_up,
                for/crp/pst/urb/irr/wet/lka_up, log_ppd_up, hft_up
 F operations   gauge storage of the 1980-2020 test set (use == True), WY1981-2009
                (1980-10-01..2009-09-30), >= 5 water years with >= 292 days:
                irreg_8109 (1 - R2 of storage vs its own 31-day-smoothed day-of-year mean cycle),
                draw_8109 (mean WY max - min / capacity), iav_8109 (sd of WY means / capacity),
                full_frac / empty_frac (share of days >= 0.95 / <= 0.10 of the 99th percentile),
                peak_sin / peak_cos (day of the cycle maximum), trend_8109 (slope of WY means,
                per decade, / capacity);
                ResOpsUS outflow/inflow WY1981-2009: rel_cv (cv of daily release), rel_in_r (r of
                the 12 monthly-mean release vs inflow values);
                two variables use gauge data of the scored window, so analyses should be reported
                with and without them: cyc_shift (1 - r between the 2010-20 and 2023-03..2026-09
                smoothed mean annual cycles of gauge storage; the 2023-2026 test set for 2023-26)
                and cyc_amp_lr (log10 amplitude ratio of those cycles)
 G SWOT quality SWOT = the storage targets in the store zarr_store_vd2026_halfexact (the values an
                earlier model configuration, a180, trained on): dv_SWOTAE at nodes of the SWOT
                node list (dv_SWOTAE_vd2026_fb_g), else dv_staticPLD that day; summed over the
                members (those of the 2023-2026 test set, else of the 1980-2020 test set) on days
                every member has a value, 2023-03-01..2026-09-30.
                log_npass (log10(1 + full-coverage SWOT days)), maxgap_d, ae_frac (share of
                member-days from dv_SWOTAE), swot_all_sup (every member on the SWOT node list),
                half_strict (share of members in HALF's tier1_strict), area_ratio_pld (area_ratio
                of the 1980-2020 sites table, |log10|);
                against the gauge of the 2023-2026 test set (truth_clean_v1x) on shared days
                (>= 10), both with their mean removed: swot_r, swot_z (Fisher z), swot_nrmse /
                swot_lnrmse (RMSE / range of the gauge over the window), swot_sd (sd ratio),
                swot_slope (OLS SWOT on gauge), swot_ndays
 H gauge quality  sites table of the 1980-2020 test set: tier_med, l1_grade (A=0, B=1, ...),
                low_signal, primary_share, n_weak, n_confl; unit_runs.csv count (n_unit_runs);
                qa_site_year 2010-20: n_bad_years (use False, flat or scale_bad), n_review,
                max_rating_resid; ResOpsUS inconsist (INCONSISTENCIES_NOTED present);
                n_flags = n_weak + n_confl + n_unit_runs + n_bad_years + (n_review > 0)
 I inflow       ERA5 ro_mean (zarr_store_ext1980) x node area, summed over the graph v2.2 upstream
                nodes present in the store, vs ResOpsUS DAILY_AV_INFLOW, 2010-20, monthly means
                (>= 20 days, >= 24 months): e5q_r, e5q_rns (r of monthly anomalies from each
                series' own monthly climatology), e5q_lratio (log10 mean ratio), e5q_lbias
                (|e5q_lratio - median over the reservoirs|), e5q_cover (area share present)
 J controls     gauge storage of the 1980-2020 test set in 2010-20 (use == True): g_shape (sd /
                range of the mean-removed gauge), g_seas (R2 of the annual harmonic + trend fit),
                log_ndays
 N negative     controls that should explain nothing: noise1/2/3, Gaussian random fields on
                lat/lon (exponential covariance, ranges 100 / 300 / 1,000 km, seed 0); iid1,
                independent Gaussian noise
"""
import os, sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json, numpy as np, pandas as pd, xarray as xr, pyogrio
import pa_lib as L

ids = L.site_ids(); S3 = L.sites(L.T3).loc[ids]; S1X = L.sites(L.T1X); g = L.Graph(); sb = g.sb
V = pd.DataFrame(index=pd.Index(ids, name="DAM_ID")); GRP, DESC = {}, {}
def put(col, grp, vals, desc=""):
    """Add one variable to the table V and record its group and description for vars_dict.csv.

    col is the column name, grp the group letter, desc the description. vals is a dict or Series
    keyed by DAM_ID (aligned to the table's index) or an array in the order of the table's rows.
    """
    V[col] = pd.Series(vals).reindex(V.index) if isinstance(vals, (dict, pd.Series)) else vals; GRP[col] = grp; DESC[col] = desc
lg = lambda x: np.log10(np.where(np.asarray(x, float) > 0, x, np.nan))
B = pd.read_csv(L.OUT/"blocks.csv").set_index("DAM_ID"); V["lat"], V["lon"] = B.lat, B.lon
print("sites", len(ids), flush=True)

# ---------------- statics and upstream sets ----------------
A = pd.read_parquet(L.STAT).reset_index(); A["subbasin"] = A.subbasin.astype(str); A = A.drop_duplicates("subbasin").set_index("subbasin")
A = A.reindex(sb.index); area = sb.area_km2.values; pos = {n: i for i, n in enumerate(sb.index)}
UP = {d: np.array(sorted(pos[x] for x in g.upstream(S3.node[d]) if x in pos)) for d in ids}
own = {d: S3.node[d] for d in ids}
def col(c, scale=1.0):
    """Return static attribute c per graph node, missing values (<= -999) set to NaN, times scale."""
    v = A[c].astype(float).values; return np.where(v <= -999, np.nan, v)*scale
def upmean(v):
    """Return the area-weighted mean of per-node array v over each reservoir's upstream nodes.

    Nodes without a finite value are left out. The result is a dict DAM_ID -> mean (NaN when no
    upstream node has a value).
    """
    out = {}
    for d in ids:
        i = UP[d]; ok = np.isfinite(v[i]); w = area[i]*ok
        out[d] = float(np.nansum(w*np.nan_to_num(v[i]))/w.sum()) if w.sum() > 0 else np.nan
    return out
def loc(v):
    """Return the value of per-node array v at each reservoir's own node, as a dict by DAM_ID.

    The value is NaN when the node is not in the graph.
    """
    return {d: v[pos[own[d]]] if own[d] in pos else np.nan for d in ids}
ai = col("ari_ix_sav", 0.01); put("ai_up", "A", upmean(ai), "aridity index P/PET, upstream"); put("ai_loc", "A", loc(ai), "aridity, own subbasin")
put("cmi_up", "A", upmean(col("cmi_ix_syr", 0.01)), "climate moisture index, upstream")
cm = np.nanmin(np.column_stack([col(f"cmi_ix_s{m:02d}", 0.01) for m in range(1, 13)]), axis=1); put("cmi_min_up", "A", upmean(cm), "driest-month CMI")
put("snow_up", "A", upmean(col("snw_pc_syr")), "snow cover %, annual, upstream"); put("snowmax_up", "A", upmean(col("snw_pc_smx")), "max monthly snow %")
Pm = {m: upmean(col(f"pre_mm_s{m:02d}")) for m in range(1, 13)}; Em = {m: upmean(col(f"pet_mm_s{m:02d}")) for m in range(1, 13)}
pse, lag = {}, {}
for d in ids:
    p = np.array([Pm[m][d] for m in range(1, 13)]); e = np.array([Em[m][d] for m in range(1, 13)])
    if np.isfinite(p).all() and p.sum() > 0:
        pse[d] = float(np.abs(p - p.sum()/12).sum()/p.sum()); k = abs(int(np.argmax(p)) - int(np.argmax(e))) if np.isfinite(e).all() else np.nan
        lag[d] = float(min(k, 12 - k)) if np.isfinite(k) else np.nan
put("pseas_up", "A", pse, "precipitation seasonality index (upstream monthly P)"); put("ppet_lag_up", "A", lag, "months between P and PET peaks")
put("tmp_up", "A", upmean(col("tmp_dc_syr", 0.1)), "air temperature degC"); put("pre_up", "A", upmean(col("pre_mm_syr")), "P mm/yr")
put("pet_up", "A", upmean(col("pet_mm_syr")), "PET mm/yr"); put("aet_ratio_up", "A", V.pre_up*np.nan, "")
aet = upmean(col("aet_mm_syr")); V["aet_ratio_up"] = pd.Series(aet)/V.pre_up; DESC["aet_ratio_up"] = "AET / P upstream"
put("run_up", "A", upmean(col("run_mm_syr")), "land surface runoff mm/yr"); put("gla_up", "A", upmean(col("gla_pc_sse")), "glacier %")
print("A done", flush=True)
# E basin
put("log_uparea", "E", lg(sb.uparea_km2.reindex([own[d] for d in ids]).values), "graph v2.2 upstream area km2")
put("ele_up", "E", upmean(col("ele_mt_sav")), "elevation m"); put("ele_loc", "E", loc(col("ele_mt_sav")), "")
emx, emn = loc(col("ele_mt_smx")), loc(col("ele_mt_smn")); put("relief_loc", "E", {d: emx[d] - emn[d] for d in ids}, "own-subbasin relief m")
put("slp_up", "E", upmean(col("slp_dg_sav", 0.1)), "terrain slope deg"); put("sgr_loc", "E", loc(col("sgr_dk_sav")), "stream gradient")
for c, k, s in (("cly_pc_sav", "cly_up", 1), ("slt_pc_sav", "slt_up", 1), ("snd_pc_sav", "snd_up", 1), ("soc_th_sav", "soc_up", 1),
                ("kar_pc_sse", "kar_up", 1), ("gwt_cm_sav", "gwt_up", 1), ("for_pc_sse", "for_up", 1), ("crp_pc_sse", "crp_up", 1),
                ("pst_pc_sse", "pst_up", 1), ("urb_pc_sse", "urb_up", 1), ("ire_pc_sse", "irr_up", 1), ("lka_pc_sse", "lka_up", 0.1),
                ("hft_ix_s09", "hft_up", 0.1)):
    put(k, "E", upmean(col(c, s)), c)
put("wet_up", "E", upmean(col("wet_pc_sg1") + col("wet_pc_sg2")), "wetland %"); put("log_ppd_up", "E", lg(list(upmean(col("ppd_pk_sav")).values())), "population density")
print("E done", flush=True)

# ---------------- C reservoir (GDW) ----------------
GD = pyogrio.read_dataframe(L.R/"GDW_v1_0/GDW_v1_0.gdb", layer="GDW_barriers_v1_0", read_geometry=False)
GD = GD[GD.GRAND_ID.notna()].copy(); GD["GRAND_ID"] = GD.GRAND_ID.astype(int); GD = GD.drop_duplicates("GRAND_ID").set_index("GRAND_ID").reindex(ids)
num = lambda c: pd.to_numeric(GD[c], errors="coerce").where(lambda v: v > -99)
cap, are = num("CAP_MCM"), num("AREA_SKM")
put("log_cap", "C", lg(cap), "GDW CAP_MCM"); put("log_area", "C", lg(are), "GDW AREA_SKM"); put("log_depth", "C", lg(cap/are), "CAP/AREA m")
put("depth_gdw", "C", num("DEPTH_M"), "GDW DEPTH_M"); put("log_hgt", "C", lg(num("DAM_HGT_M")), "dam height")
put("age", "C", 2015 - num("YEAR_DAM"), "years in 2015"); put("log_dor_gdw", "C", lg(num("DOR_PC")), "GDW DOR_PC %")
put("elev_dam", "C", num("ELEV_MASL"), ""); put("log_power", "C", np.log10(1 + num("POWER_MW").fillna(0)), "")
uses = ["USE_IRRI", "USE_ELEC", "USE_SUPP", "USE_FCON", "USE_RECR", "USE_NAVI"]
has = lambda c: GD[c].notna() & ~GD[c].astype(str).str.strip().isin(["", "None", "nan", "No"])
for u in uses: put(u.lower(), "C", has(u).astype(float).values, u)
put("n_uses", "C", sum(has(u).astype(float) for u in uses + ["USE_FISH", "USE_PCON", "USE_LIVE", "USE_OTHR"]).values, "number of uses")
mu = GD.MAIN_USE.astype(str)
for k, lab in (("irri", "Irrigation"), ("elec", "Hydroelectricity"), ("supp", "Water supply"), ("fcon", "Flood control"), ("recr", "Recreation"), ("navi", "Navigation")):
    put(f"main_{k}", "C", (mu == lab).astype(float).values, f"MAIN_USE == {lab}")
put("lake_ctrl", "C", has("LAKE_CTRL").astype(float).values, "regulated natural lake (LAKE_CTRL)")
put("n_members", "C", [len(S3.mem[d]) for d in ids], "PLD member lakes")
put("log_cap_per_uparea", "C", lg(cap.values/sb.uparea_km2.reindex([own[d] for d in ids]).values), "capacity per upstream km2")
Q = pd.read_csv(L.F/"ResOpsUS/time_series_single_variable_table/DAILY_AV_INFLOW_CUMECS.csv", usecols=lambda c: c == "date" or (c.isdigit() and int(c) in set(ids)))
Q["date"] = pd.to_datetime(Q.date); Q = Q.set_index("date"); Q = Q[~Q.index.duplicated()].sort_index(); wyf = lambda ix: ix.year + (ix.month >= 10); vq = {}
for c in Q.columns:
    q = Q[c].dropna(); n = q.groupby(wyf(q.index)).size(); good = n[n >= 292].index
    if len(good) >= 5:
        m = q[wyf(q.index).isin(good)].mean()
        if m > 0 and np.isfinite(cap.get(int(c), np.nan)): vq[int(c)] = cap[int(c)]/(m*365.25*86400/1e6)
put("log_vq_obs", "C", lg(pd.Series(vq).reindex(ids).values), "capacity / ResOpsUS mean inflow, years")
print("C done", flush=True)

# ---------------- D regulation ----------------
dc = col("dam_cap_mcm"); isd = np.isfinite(dc) & (dc > 0); fb = set(pd.read_csv(L.FB).subbasin.astype(str)); isf = np.array([n in fb for n in sb.index])
dis = col("dis_m3_pyr"); upd, upc, upf, ndn, udor = {}, {}, {}, {}, {}
for d in ids:
    i = UP[d]; mem = {pos[m] for m in S3.mem[d] if m in pos} | {pos[own[d]]}; j = np.array([x for x in i if x not in mem], dtype=int)
    upd[d] = int(isd[j].sum()) if len(j) else 0; upc[d] = float(np.nansum(dc[j][isd[j]])) if len(j) else 0.0; upf[d] = int(isf[j].sum()) if len(j) else 0
    q = dis[pos[own[d]]]; udor[d] = upc[d]/(q*365.25*86400/1e6) if np.isfinite(q) and q > 0 else np.nan
    ndn[d] = int(sum(isd[pos[x]] for x in g.down_path(own[d]) if x in pos))
put("log1p_up_dams", "D", np.log10(1 + pd.Series(upd).reindex(ids).values), "GDW dam nodes upstream"); put("log_up_cap", "D", np.log10(1 + pd.Series(upc).reindex(ids).values), "")
put("up_dor_cum", "D", udor, "upstream capacity / mean annual flow"); put("n_down_res", "D", ndn, "dam nodes downstream")
put("log1p_up_swot", "D", np.log10(1 + pd.Series(upf).reindex(ids).values), "SWOT-supervised lake nodes upstream")
put("log_dor_ha", "D", np.log10(1 + np.array(list(loc(col("dor_pc_pva")).values()))), "HydroATLAS dor_pc_pva")
GC = pd.read_csv(L.F/"eval/a180_graphdiff/classes.csv").set_index("DAM_ID").reindex(ids)
put("graph_changed", "D", GC.cls.isin(["G-minor", "G-major"]).astype(float).values, "v1->v2.2 graph change"); put("graph_fixed", "D", GC.fixed.astype(float).values, "")
put("area_err_v22", "D", GC.err_merit_v22.astype(float).abs().values, "|err| v2.2 upstream area vs MERIT")
put("catch_lratio", "D", np.abs(lg(num("CATCH_SKM").values/sb.uparea_km2.reindex([own[d] for d in ids]).values)), "|log10 GDW catchment / graph area|")
print("D done", flush=True)

# ---------------- F operations, J controls (truth only) ----------------
TR = pd.read_parquet(L.T3/"truth_clean.parquet"); TR["date"] = pd.to_datetime(TR.date); TR = TR[TR.use.fillna(False).astype(bool) & TR.DAM_ID.isin(ids)]
T26 = pd.read_parquet(L.T1X/"truth_clean_v1x.parquet"); T26["date"] = pd.to_datetime(T26.date)
def cycle(s):
    """Return the mean seasonal cycle of a daily series, indexed by day of year 1-365.

    The cycle is the day-of-year mean (day 366 folded into 365), with gaps interpolated and a 31-day
    circular running mean applied. Return None when fewer than 330 days of the year have data.
    """
    doy = np.minimum(s.index.dayofyear.values, 365); m = pd.Series(s.values).groupby(doy).mean().reindex(range(1, 366))
    if m.notna().sum() < 330: return None
    m = m.interpolate(limit_direction="both"); e = np.r_[m.values[-15:], m.values, m.values[:15]]
    return pd.Series(np.convolve(e, np.ones(31)/31, "valid"), index=range(1, 366))
F_ = {k: {} for k in ("irreg_8109", "draw_8109", "iav_8109", "full_frac", "empty_frac", "peak_sin", "peak_cos", "trend_8109", "cyc_shift", "cyc_amp_lr",
                      "g_shape", "g_seas", "log_ndays")}
O26 = {int(k): g_.set_index("date").storage_mcm.sort_index() for k, g_ in T26[T26.DAM_ID.isin(ids)].groupby("DAM_ID")}
for d, g_ in TR.groupby("DAM_ID"):
    s = g_.set_index("date").storage_mcm.sort_index(); c_ = cap.get(d, np.nan)
    a = s[(s.index >= "1980-10-01") & (s.index < "2009-10-01")]; wy = wyf(a.index); n = a.groupby(wy).size(); good = n[n >= 292].index
    if len(good) >= 5:
        a = a[wy.isin(good)]; cy = cycle(a)
        if cy is not None:
            an = a - a.mean(); cv = cy.reindex(np.minimum(a.index.dayofyear, 365)).values; cv = cv - cv.mean()
            F_["irreg_8109"][d] = float(((an.values - cv)**2).sum()/(an.values**2).sum()); pk = int(cy.idxmax())
            F_["peak_sin"][d], F_["peak_cos"][d] = np.sin(2*np.pi*pk/365), np.cos(2*np.pi*pk/365)
        gw = a.groupby(wyf(a.index)); p99 = a.quantile(0.99)
        if np.isfinite(c_) and c_ > 0:
            F_["draw_8109"][d] = float((gw.max() - gw.min()).mean()/c_); F_["iav_8109"][d] = float(gw.mean().std()/c_)
            yy = gw.mean(); F_["trend_8109"][d] = float(np.polyfit(yy.index.values, yy.values, 1)[0]*10/c_)
        if p99 > 0: F_["full_frac"][d] = float((a >= 0.95*p99).mean()); F_["empty_frac"][d] = float((a <= 0.10*p99).mean())
    h = s[(s.index >= L.W0) & (s.index <= L.W1)]
    if len(h) >= 730:
        oc = h - h.mean(); t = (h.index - h.index[0]).days.values.astype(float); Wh = 2*np.pi/365.25
        X = np.column_stack([np.ones_like(t), t/365.25, np.cos(Wh*t), np.sin(Wh*t)]); b = np.linalg.lstsq(X, oc.values, rcond=None)[0]
        F_["g_shape"][d] = float(oc.std(ddof=0)/(oc.max() - oc.min())); F_["g_seas"][d] = float(1 - ((oc.values - X @ b)**2).sum()/(oc.values**2).sum())
        F_["log_ndays"][d] = float(np.log10(len(h))); c1 = cycle(h)
        if d in O26 and c1 is not None:
            c2 = cycle(O26[d][(O26[d].index >= L.X0) & (O26[d].index <= L.X1)])
            if c2 is not None:
                F_["cyc_shift"][d] = float(1 - np.corrcoef(c1, c2)[0, 1]); a1, a2 = c1.max() - c1.min(), c2.max() - c2.min()
                F_["cyc_amp_lr"][d] = float(np.log10(a2/a1)) if a1 > 0 and a2 > 0 else np.nan
for k, v in F_.items(): put(k, "J" if k in ("g_shape", "g_seas", "log_ndays") else "F", v, "")
OF = pd.read_csv(L.F/"ResOpsUS/time_series_single_variable_table/DAILY_AV_OUTFLOW_CUMECS.csv", usecols=lambda c: c == "date" or (c.isdigit() and int(c) in set(ids)))
OF["date"] = pd.to_datetime(OF.date); OF = OF.set_index("date"); OF = OF[~OF.index.duplicated()].sort_index(); rcv, rir = {}, {}
for c in OF.columns:
    o = OF[c][(OF.index >= "1980-10-01") & (OF.index < "2009-10-01")].dropna(); n = o.groupby(wyf(o.index)).size(); good = n[n >= 292].index
    if len(good) < 5: continue
    o = o[wyf(o.index).isin(good)]
    if o.mean() > 0: rcv[int(c)] = float(o.std()/o.mean())
    if c in Q.columns:
        i_ = Q[c].reindex(o.index); ok = i_.notna()
        if ok.sum() > 730:
            mo, mi = o[ok].groupby(o[ok].index.month).mean(), i_[ok].groupby(o[ok].index.month).mean()
            if len(mo) == 12 and mo.std() > 0 and mi.std() > 0: rir[int(c)] = float(np.corrcoef(mo, mi)[0, 1])
put("rel_cv", "F", rcv, "cv of daily release WY1981-2009"); put("rel_in_r", "F", rir, "r monthly release vs inflow climatology")
print("F/J done", flush=True)

# ---------------- H truth quality ----------------
cnt = lambda v: 0 if pd.isna(v) or str(v).strip() == "" else len([x for x in str(v).split(";") if x.strip()])
put("tier_med", "H", (S3.site_tier == "Medium").astype(float).values, "ResOpsUS tier Medium")
put("l1_grade", "H", S3.L1_grade.map({"A": 0, "B": 1, "C": 2, "D": 3}).astype(float).values, "L1 grade A=0")
put("low_signal", "H", S3.low_signal.astype(float).values, ""); put("primary_share", "H", S3.primary_share.astype(float).values, "")
put("n_weak", "H", [cnt(v) for v in S3.weaknesses], ""); put("n_confl", "H", [cnt(v) for v in S3.conflicts], "")
UR = pd.read_csv(L.T3/"unit_runs.csv"); put("n_unit_runs", "H", UR.groupby("DAM_ID").size().reindex(ids).fillna(0).values, "")
QA = pd.read_csv(L.T3/"qa_site_year.csv"); QA = QA[(QA.year >= 2010) & (QA.year <= 2020) & QA.DAM_ID.isin(ids)]
bad = (~QA.use.astype(bool)) | QA.flat.astype(bool) | QA.scale_bad.astype(bool)
put("n_bad_years", "H", bad.groupby(QA.DAM_ID).sum().reindex(ids).fillna(0).values, "2010-20 site-years not used / flat / scale_bad")
put("n_review", "H", QA.groupby("DAM_ID").n_flag_review.sum().reindex(ids).fillna(0).values, "")
put("max_rating_resid", "H", QA.assign(a=QA.rating_resid.abs()).groupby("DAM_ID").a.max().reindex(ids).values, "")
att = pd.read_csv(L.F/"ResOpsUS/attributes/reservoir_attributes.csv").set_index("DAM_ID").reindex(ids)
put("inconsist", "H", att.INCONSISTENCIES_NOTED.notna().astype(float).values, "ResOpsUS inconsistency note")
V["n_flags"] = V.n_weak + V.n_confl + V.n_unit_runs + V.n_bad_years + (V.n_review > 0); GRP["n_flags"] = "H"; DESC["n_flags"] = "sum of truth flags"
print("H done", flush=True)

# ---------------- G SWOT quality (no model output) ----------------
mem26 = {d: (S1X.mem[d], S1X.basin[d]) if d in S1X.index else (S3.mem[d], S3.basin[d]) for d in ids}
HL = pd.read_csv(L.F/"eval/half_vd_review/halfexact_lakes_vD.csv", dtype={"lake_id": str}).set_index("lake_id")
G_ = {k: {} for k in ("log_npass", "maxgap_d", "ae_frac", "swot_all_sup", "half_strict", "swot_r", "swot_z", "swot_nrmse", "swot_lnrmse", "swot_sd", "swot_slope", "swot_ndays")}
for bas in sorted({b for _, b in mem26.values()}):
    ds = xr.open_zarr(L.STORE_SWOT/bas, consolidated=False); have = set(ds.subbasin.values.astype(str))
    need = sorted({m for d, (mm, b) in mem26.items() if b == bas for m in mm} & have)
    if not need: continue
    sub = ds[["dv_SWOTAE", "dv_staticPLD"]].sel(subbasin=need, date=slice(L.X0, L.X1)).load()
    ae = sub.dv_SWOTAE.to_pandas(); sp = sub.dv_staticPLD.to_pandas(); ae.columns = ae.columns.astype(str); sp.columns = sp.columns.astype(str)
    for d, (mm, b) in mem26.items():
        if b != bas: continue
        G_["swot_all_sup"][d] = float(all(m in fb for m in mm)); G_["half_strict"][d] = float(np.mean([HL.tier.get(m) == "tier1_strict" for m in mm]))
        if not all(m in ae.columns for m in mm): continue
        v = pd.DataFrame({m: ae[m].where(ae[m].notna(), sp[m]) if m in fb else np.nan for m in mm}, index=ae.index)
        ok = v.notna().all(axis=1); s = v[ok].sum(axis=1)
        G_["log_npass"][d] = float(np.log10(1 + len(s)))
        if not len(s): continue
        G_["log_npass"][d] = float(np.log10(1 + len(s))); G_["maxgap_d"][d] = float(np.diff(np.r_[L.X0.value, s.index.values.astype("int64"), L.X1.value]).max()/864e11)
        G_["ae_frac"][d] = float(np.mean([ae[m][ok].notna().mean() for m in mm]))
        if d in O26:
            o = O26[d][(O26[d].index >= L.X0) & (O26[d].index <= L.X1)]; j = pd.concat([s.rename("s"), o.rename("g")], axis=1).dropna()
            G_["swot_ndays"][d] = len(j)
            if len(j) >= 10 and j.s.std() > 0 and j.g.std() > 0:
                sc, gc = j.s - j.s.mean(), j.g - j.g.mean(); rg = o.max() - o.min(); r = float(np.corrcoef(sc, gc)[0, 1])
                G_["swot_r"][d] = r; G_["swot_z"][d] = float(np.arctanh(np.clip(r, -0.999, 0.999)))
                G_["swot_nrmse"][d] = float(np.sqrt(((sc - gc)**2).mean())/rg) if rg > 0 else np.nan; G_["swot_lnrmse"][d] = np.log10(G_["swot_nrmse"][d])
                G_["swot_sd"][d] = float(sc.std()/gc.std()); G_["swot_slope"][d] = float((sc*gc).sum()/(gc**2).sum())
    print("G", bas, flush=True)
for k, v in G_.items(): put(k, "G", v, "")
put("area_ratio_pld", "G", np.abs(lg(S3.area_ratio.astype(float).values)), "|log10 PLD/GDW area ratio|")

# ---------------- B climate shift (longmem_vd2026, ERA5-derived) ----------------
Bv = {k: {} for k in ("shift_spei12", "shift_ssi12", "abs_shift_spei12", "out_lk24", "out_lk6", "dry_frac", "wet_frac")}
for bas in sorted(S3.basin.unique()):
    p = L.LM/bas
    if not p.exists(): continue
    ds = xr.open_zarr(p, consolidated=False); have = set(ds.subbasin.values.astype(str)); dd = [d for d in ids if S3.basin[d] == bas and own[d] in have]
    if not dd: continue
    x = ds[["lm_up_spei12", "lm_up_ssi12", "lm_up_lk24", "lm_up_lk6"]].sel(subbasin=[own[d] for d in dd]).load().astype("float32")
    t = pd.DatetimeIndex(x.date.values); h = (t >= L.W0) & (t <= L.W1); w = (t >= L.X0) & (t <= L.X1)
    for d in dd:
        z = {k: x[k].sel(subbasin=own[d]).values for k in x.data_vars}
        Bv["shift_spei12"][d] = float(np.nanmean(z["lm_up_spei12"][h]) - np.nanmean(z["lm_up_spei12"][w])); Bv["abs_shift_spei12"][d] = abs(Bv["shift_spei12"][d])
        Bv["shift_ssi12"][d] = float(np.nanmean(z["lm_up_ssi12"][h]) - np.nanmean(z["lm_up_ssi12"][w]))
        for k, c in (("out_lk24", "lm_up_lk24"), ("out_lk6", "lm_up_lk6")):
            a, b = np.nanmin(z[c][w]), np.nanmax(z[c][w]); hv = z[c][h]; hv = hv[np.isfinite(hv)]; Bv[k][d] = float(((hv < a) | (hv > b)).mean()) if len(hv) else np.nan
        sp = z["lm_up_spei12"][h]; sp = sp[np.isfinite(sp)]; Bv["dry_frac"][d] = float((sp < -1.5).mean()); Bv["wet_frac"][d] = float((sp > 1.5).mean())
for k, v in Bv.items(): put(k, "B", v, "")
print("B done", flush=True)

# ---------------- I inflow quality (ERA5 runoff vs ResOpsUS inflow) ----------------
Iv = {k: {} for k in ("e5q_r", "e5q_rns", "e5q_lratio", "e5q_cover")}
Qh = Q[(Q.index >= L.W0) & (Q.index <= L.W1)]
for bas in sorted(S3.basin.unique()):
    dd = [d for d in ids if S3.basin[d] == bas and str(d) in Qh.columns and Qh[str(d)].notna().sum() >= 600]
    if not dd: continue
    ds = xr.open_zarr(L.STORE_EXT/bas, consolidated=False); have = set(ds.subbasin.values.astype(str))
    nodes = sorted({sb.index[i] for d in dd for i in UP[d]} & have)
    ro = ds.ro_mean.sel(subbasin=nodes, date=slice(L.W0, L.W1)).load().to_pandas(); ro.columns = ro.columns.astype(str)
    vol = ro.mul(sb.area_km2.reindex(ro.columns).values, axis=1)
    for d in dd:
        un = [sb.index[i] for i in UP[d] if sb.index[i] in have]; Iv["e5q_cover"][d] = float(sb.area_km2.reindex(un).sum()/area[UP[d]].sum())
        e = vol[un].sum(axis=1); o = Qh[str(d)]; j = pd.concat([e.rename("e"), o.rename("o")], axis=1).dropna()
        mo = j.groupby(j.index.to_period("M")).agg(["mean", "size"]); mo = mo[mo[("o", "size")] >= 20]
        if len(mo) < 24: continue
        me, mob = mo[("e", "mean")], mo[("o", "mean")]; Iv["e5q_r"][d] = float(np.corrcoef(me, mob)[0, 1])
        mon = me.index.month; ae_, ao_ = me - me.groupby(mon).transform("mean"), mob - mob.groupby(mon).transform("mean")
        Iv["e5q_rns"][d] = float(np.corrcoef(ae_, ao_)[0, 1]) if ae_.std() > 0 and ao_.std() > 0 else np.nan
        Iv["e5q_lratio"][d] = float(np.log10(me.mean()/mob.mean())) if me.mean() > 0 and mob.mean() > 0 else np.nan
    print("I", bas, len(dd), flush=True)
for k, v in Iv.items(): put(k, "I", v, "")
V["e5q_lbias"] = (V.e5q_lratio - V.e5q_lratio.median()).abs(); GRP["e5q_lbias"] = "I"; DESC["e5q_lbias"] = "|log10 ERA5/obs inflow ratio - median|"

# ---------------- N negative controls ----------------
rng = np.random.default_rng(0); la, lo = np.radians(V.lat.values), np.radians(V.lon.values)
dlat, dlon = la[:, None] - la[None], lo[:, None] - lo[None]
Dk = 2*6371*np.arcsin(np.sqrt(np.sin(dlat/2)**2 + np.cos(la[:, None])*np.cos(la[None])*np.sin(dlon/2)**2))
for k, rr in (("noise1", 100), ("noise2", 300), ("noise3", 1000)):
    Cm = np.exp(-Dk/rr) + 1e-6*np.eye(len(V)); put(k, "N", np.linalg.cholesky(Cm) @ rng.standard_normal(len(V)), f"GRF range {rr} km")
put("iid1", "N", rng.standard_normal(len(V)), "iid noise")

# ---------------- write ----------------
V = V.astype(float); V.to_parquet(L.OUT/"vars.parquet")
DD = pd.DataFrame({"group": pd.Series(GRP), "desc": pd.Series(DESC)}); DD["n_valid"] = V.notna().sum().reindex(DD.index); DD.to_csv(L.OUT/"vars_dict.csv")
REP = {"A": ["ai_up", "snow_up", "pseas_up"], "B": ["shift_spei12", "out_lk24", "dry_frac"], "C": ["log_cap", "log_dor_gdw", "log_depth"],
       "D": ["log1p_up_dams", "up_dor_cum"], "E": ["log_uparea", "ele_up", "irr_up"], "F": ["irreg_8109", "draw_8109", "cyc_shift"],
       "G": ["swot_z", "swot_lnrmse", "log_npass"], "H": ["tier_med", "primary_share", "n_flags"], "I": ["e5q_r", "e5q_lbias"],
       "J": ["g_shape", "g_seas", "log_ndays"], "N": ["noise1", "noise2", "noise3", "iid1"]}
FALLBACK = {"log_dor_gdw": "log_vq_obs", "swot_z": "log_npass"}
note = {}
for gk, cols in REP.items():
    for i, c in enumerate(cols):
        if V[c].isna().mean() > 0.6:
            fbk = FALLBACK.get(c); note[c] = f"{V[c].notna().sum()} valid; " + (f"replaced by {fbk}" if fbk else "kept (no fallback)")
            if fbk and fbk not in cols: cols[i] = fbk
json.dump({"reps": REP, "coverage_notes": note, "outcome_adjacent": ["cyc_shift", "cyc_amp_lr"]}, open(L.OUT/"reps.json", "w"), indent=1)
print(DD.groupby("group").size().to_dict(), "columns", V.shape)
print(DD.sort_values("n_valid").head(25).to_string())
print("rep coverage:", {c: int(V[c].notna().sum()) for cs in REP.values() for c in cs}, note)
