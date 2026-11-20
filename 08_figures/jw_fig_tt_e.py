"""Load the daily series behind Figure 1 (and, below the split point, draw an earlier eight-row figure).

Only the first part of this file is used by the paper's figures: fig1_c5d.py (and combo_e.py) read this file
as text, cut it at the line that creates the figure, replace the list of example reservoirs (EX) and execute
that first part. It defines the loaders and score helpers Figure 1 needs. The drawing code after the split
point belongs to an earlier figure layout and is not run. Do not change the EX line or the figure-creation line.

What the first part provides
  * hindcast series per reservoir: gauge truth (testsets/regime3_full_1980_2020), SWOT-GAT (label `SWOT-GNN`
    or `v10pm` in code: the mean of the two seeds), and the three rule models started from the gauge;
  * SWOT Period series: gauge truth (testsets/regime1x_2023_2026), SWOT-GAT, the rule-model runs to June 2026,
    and the SWOT observations the model trained on (the lake record after the HALF filter), placed on the
    gauge's level, since SWOT measures storage change;
  * scores(): nRMSE, NSE and the variability ratio (VR) on daily values of mean-removed series;
  * PCT: each reservoir's percentile of 2010-2020 hindcast NSE among reservoirs all four models score.

Display conventions used by Figure 1: daily storage anomaly in million cubic metres, each series centred on
its own mean over the days it shares with the gauge; lines are 7-day running means for display only, scores
use daily values; the SWOT Period panels show and score each reservoir's own SWOT span (first to last pass).
"""
from swotgat.paths import WORK
import os, sys; sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import json, numpy as np, pandas as pd, matplotlib as mpl
import jw; import grl_common as G, grl2 as K
import matplotlib.patheffects as pe
from matplotlib.colors import to_rgb
LINEB = "#2a78d6"                                         # SWOT-GNN blue used across the set
tint = lambda c, f=0.45: tuple(1 - (1 - np.array(to_rgb(c)))*(1 - f))
G.SERIES_FILE = __import__("ipcfg").SERIES                        # gauge-started benchmark runs, as in every statistic (v10pm is identical in both files)
RULES = os.environ.get("RULES", "0") == "1"; RL = ["MODROM-cal", "GDROM", "ISTARF"]
RCLR = {"GDROM": "#eb6834", "ISTARF": "#1baf7a", "MODROM-cal": "#eda100"}
from jw import plt
from pathlib import Path
jw.use_style(); F = Path(f"{WORK}/CONUS_storage_fork"); OUT = Path(__file__).resolve().parent
EX = [144, 1275, 1042, 924, 131, 965, 601, 1035]          # tt_candidates.py; Glen Canyon 597 -> Navajo 601, El Vado 604 -> Eucha 1035 (no HALF-exact SWOT; lmdft_vDspan_20261019/fig4_swaps.py)
W0, W1 = pd.Timestamp("2010-01-01"), pd.Timestamp("2020-12-31"); PR = F/"predictions"; T3 = F/"testsets/regime3_full_1980_2020"
COLA = {"s3": jw.ARCH["sky"], "s0": jw.ARCH["teal"], "s1": jw.ARCH["navy"], "s2": jw.ARCH["pink"], "carryover": jw.ARCH["orange"]}
CODE = {"s3": "A1", "s0": "A2", "s1": "A3", "s2": "A4", "carryover": "A5"}
USE = {"Hydroelectricity": "hydropower", "Irrigation": "irrigation", "Flood control": "flood control", "Water supply": "water supply", "Navigation": "navigation", "Recreation": "recreation"}
SWOT_RING = "#d28a2a"
ST3 = pd.read_csv(T3/"sites.csv", dtype={"node": str, "members": str, "basin": str}).set_index("DAM_ID")
TR3 = pd.read_parquet(T3/"truth_clean.parquet"); TR3["date"] = pd.to_datetime(TR3.date)
TR3 = TR3[TR3.use.fillna(False).astype(bool) & (TR3.date >= W0) & (TR3.date <= W1) & TR3.DAM_ID.isin(EX)]
need = {(str(ST3.basin[d_]), m_) for d_ in EX for m_ in str(ST3.members[d_]).split(";")}
def gnn_run(run):
    """Return one run's SWOT-GAT hindcast predictions at the member nodes of the EX reservoirs.

    run is the name of a prediction run (one seed). The prediction files of the tags in ipcfg.HIND are
    read, cut to the window W0..W1 and to the (basin, node) pairs in `need`, and joined; where two files
    hold the same node and day, the first is kept. Returns the predicted storage (pred_dv_SWOTAE) as a
    Series indexed by basin, subbasin and date.
    """
    parts = []
    for c in __import__("ipcfg").HIND:
        d_ = pd.read_parquet(PR/run/f"{__import__('ipcfg').STEP[run]}_{c}.parquet", columns=["basin", "subbasin", "date", "pred_dv_SWOTAE"])
        d_["basin"] = d_.basin.astype(str); d_["subbasin"] = d_.subbasin.astype(str); d_["date"] = pd.to_datetime(d_.date)
        d_ = d_[(d_.date >= W0) & (d_.date <= W1)]; parts.append(d_[[(b, s_) in need for b, s_ in zip(d_.basin, d_.subbasin)]])
    return pd.concat(parts).drop_duplicates(["basin", "subbasin", "date"]).set_index(["basin", "subbasin", "date"]).pred_dv_SWOTAE
GN = pd.concat([gnn_run(r_) for r_ in __import__("ipcfg").RUNS], axis=1).mean(axis=1, skipna=False).rename("p").sort_index()
RUL = {}
for m_, tag in (("GDROM", "gdrom3c_g_y1980"), ("ISTARF", "istarf3c_g_y1980"), ("MODROM-cal", "modromcal3c_g_y1980")):
    d_ = pd.read_parquet(PR/f"grades_{tag}"/"full_series.parquet"); d_["date"] = pd.to_datetime(d_.date); d_ = d_[(d_.date >= W0) & (d_.date <= W1) & d_.DAM_ID.isin(EX)]
    RUL[m_] = {int(k_): g_.set_index("date").storage_mcm for k_, g_ in d_.groupby("DAM_ID")}
# ---- tt26: 2023-26 right-hand panels (kit/tt26.py) ----
import xarray as xr
T26 = pd.date_range("2023-01-01", "2026-09-30", freq="D")
X1 = F/"testsets/regime1x_2023_2026"; S1X = pd.read_csv(X1/"sites.csv", dtype={"node": str, "members": str, "basin": str}).set_index("DAM_ID")
TX = pd.read_parquet(X1/"truth_clean_v1x.parquet"); TX["date"] = pd.to_datetime(TX.date); TX = TX[TX.DAM_ID.isin(EX)]
SPX = pd.read_csv(F/"eval/score2026/spans_2023_2026.csv").set_index("DAM_ID")
def _x26(run):
    """Return one run's SWOT Period predictions (file tag x26) at the EX reservoirs' member nodes."""
    d_ = pd.read_parquet(PR/run/f"{__import__('ipcfg').STEP[run]}_x26.parquet", columns=["basin", "subbasin", "date", "pred_dv_SWOTAE"])
    d_["basin"] = d_.basin.astype(str); d_["subbasin"] = d_.subbasin.astype(str); d_["date"] = pd.to_datetime(d_.date)
    nd = {(str(S1X.basin[x]), m) for x in EX if x in S1X.index for m in str(S1X.members[x]).split(";")}
    d_ = d_[[(b, s) in nd for b, s in zip(d_.basin, d_.subbasin)]]
    return d_.drop_duplicates(["basin", "subbasin", "date"]).set_index(["basin", "subbasin", "date"]).pred_dv_SWOTAE
G26 = pd.concat([_x26(r_) for r_ in __import__("ipcfg").RUNS], axis=1).mean(axis=1, skipna=False).rename("p").sort_index()
def x26_pair(dam):
    """Return the gauge and SWOT-GAT daily storage of one reservoir on T26 (the SWOT Period).

    Returns (gauge, model) as float arrays in MCM. The model value is the two-seed mean summed over the
    reservoir's member nodes, on days when every member has a value; it is all NaN when the reservoir is
    not in the SWOT Period test set.
    """
    o = TX[TX.DAM_ID == dam].set_index("date").storage_mcm.reindex(T26).values.astype(float)
    if dam not in S1X.index: return o, np.full(len(T26), np.nan)
    mem = str(S1X.members[dam]).split(";"); g_ = G26.loc[str(S1X.basin[dam])]
    g_ = g_[g_.index.get_level_values(0).isin(mem)].reset_index(); k_ = g_.groupby("date").subbasin.nunique()
    return o, g_.groupby("date").p.sum()[k_ == len(mem)].reindex(T26).values.astype(float)
def span26(dam):
    """Return a boolean mask on T26 that is True inside the reservoir's SWOT span.

    The span runs from the first to the last SWOT pass (spans_2023_2026.csv) and ends no later than
    12 June 2026. The mask is all False when the reservoir has no span.
    """
    s = np.zeros(len(T26), bool)
    if dam in SPX.index and pd.notna(SPX.vD_start[dam]):
        s[(T26 >= pd.Timestamp(SPX.vD_start[dam])) & (T26 <= min(pd.Timestamp(SPX.vD_end[dam]), pd.Timestamp("2026-06-12")))] = True
    return s
LAST = pd.Timestamp("2026-06-12")                    # end of the 2026 rule-model runs = end of every SWOT Period panel
R26 = {}
for m_, tag in (("GDROM", "gdrom3c_g26h0"), ("ISTARF", "istarf3c_g26h0"), ("MODROM-cal", "modromcal3c_g26h0")):
    d_ = pd.read_parquet(PR/f"grades_{tag}"/"full_series.parquet"); d_["date"] = pd.to_datetime(d_.date)
    d_ = d_[(d_.date >= T26[0]) & (d_.date <= LAST) & d_.DAM_ID.isin(EX)]
    R26[m_] = {int(k_): g_.set_index("date").storage_mcm.astype(float).reindex(T26).values for k_, g_ in d_.groupby("DAM_ID")}
def rule26(m, dam):
    """Return the gauge and rule model m's daily storage of one reservoir on T26, in MCM.

    Returns (None, None) when the model has no SWOT Period run for the reservoir.
    """
    if dam not in R26[m]: return None, None
    return TX[TX.DAM_ID == dam].set_index("date").storage_mcm.reindex(T26).values.astype(float), R26[m][dam]
def rules_ok26(dam):
    """Return a boolean mask on T26 of the days on which every rule model has a value.

    Only the rule models that have this reservoir count. This is the same-days rule of the span table.
    """
    k = np.ones(len(T26), bool)
    for m in R26:
        if dam in R26[m]: k &= np.isfinite(R26[m][dam])
    return k
_mem = {m for x in EX if x in S1X.index for m in str(S1X.members[x]).split(";")}
HE = pd.read_parquet(F/"eval/half_vd_review/lakesp_vD_halfexact_graph.parquet", columns=["lake_id", "date", "dv_SWOTAE", "dv_3DLAKES", "dv_staticPLD"])
HE["lake_id"] = HE.lake_id.astype(str); HE = HE[HE.lake_id.isin(_mem)]; HE["date"] = pd.to_datetime(HE.date)
HE["v"] = HE.dv_SWOTAE.fillna(HE.dv_3DLAKES).fillna(HE.dv_staticPLD); HEV = {k_: g_.set_index("date").v.dropna() for k_, g_ in HE.groupby("lake_id")}
def swot26(dam):
    """Return the reservoir's SWOT storage from the HALF-exact record, summed over its member lakes.

    Each lake's value is dv_SWOTAE, else 3D-LAKES, else static PLD (the way the training store builds its
    target). Members are summed on days when every member has a value, and the series is cut to
    2023-07-01 .. LAST. Returns None when the reservoir is not in the SWOT Period test set or a member
    has no HALF-exact series.
    """
    if dam not in S1X.index: return None
    mem = str(S1X.members[dam]).split(";")
    if any(m not in HEV for m in mem): return None
    v = pd.concat([HEV[m] for m in mem], axis=1).dropna().sum(axis=1)
    return v[(v.index >= "2023-07-01") & (v.index <= LAST)]
def hind(dam):
    """Return the hindcast series of one reservoir as a daily table over W0..W1, in MCM.

    Columns: "gauge" (gauge storage), "SWOT-GNN" (SWOT-GAT: the two-seed mean summed over the member
    nodes, on days when every member has a value) and one column per rule model that has the reservoir.
    """
    mem = str(ST3.members[dam]).split(";"); g_ = GN.loc[str(ST3.basin[dam])].loc[mem].reset_index(); k_ = g_.groupby("date").subbasin.nunique()
    cols = {"gauge": TR3[TR3.DAM_ID == dam].set_index("date").storage_mcm, "SWOT-GNN": g_.groupby("date").p.sum()[k_ == len(mem)]}
    cols.update({m_: RUL[m_][dam] for m_ in RL if dam in RUL[m_]})
    return pd.DataFrame(cols).reindex(pd.date_range(W0, W1, freq="D"))
S10, HK, *_ = jw.sites_table("2010"); S10 = S10[S10.tier.isin(["High", "Medium"]) & S10.nrmse.notna()]
c10 = set.intersection(*[set(S10[(S10.PERIOD == HK) & (S10.MODEL == m_)].DAM_ID) for m_ in ["v10pm", "MODROM-cal", "GDROM", "ISTARF"]])
_h = S10[(S10.PERIOD == HK) & (S10.MODEL == "v10pm") & S10.DAM_ID.isin(c10)].set_index("DAM_ID"); PCT = 100*_h.nse.rank(pct=True)   # c5_o50: direct NSE column
ECO = pd.read_csv(OUT/"eco_l1_sites.csv").set_index("DAM_ID").egroup
ECN = {"ETF": "Eastern Temperate Forests", "GP": "Great Plains", "NWFM": "Northwestern Forested Mountains", "NAD": "North American Deserts", "PAC": "Pacific coast"}
ordn = lambda n: f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"
meta = pd.read_csv(F/"eval/grl_figs/sites_v5_g.csv").drop_duplicates("DAM_ID").set_index("DAM_ID")
def scores(o, p):
    """Return nRMSE, NSE and the variability ratio of p against o on the shared daily values.

    o is the gauge series and p the model series (Figure 1 passes SWOT-GAT). Both are centred on their
    mean over the days both are finite. nRMSE = RMSE / gauge range; variability ratio (VR) = model sd /
    gauge sd; NSE = 2*VR*r - VR^2, the NSE of the centred series. Returns (nRMSE, NSE, VR).
    """
    k = np.isfinite(o) & np.isfinite(p); oc, pc = o[k] - o[k].mean(), p[k] - p[k].mean(); a = pc.std()/oc.std(); r = np.corrcoef(oc, pc)[0, 1]
    return np.sqrt(np.mean((pc - oc)**2))/(oc.max() - oc.min()), 2*a*r - a*a, a
def nrmse(o, p):
    """Return the nRMSE of p against o: RMSE of the centred series on their shared days / range of o."""
    k = np.isfinite(o) & np.isfinite(p); o, p = o[k] - o[k].mean(), p[k] - p[k].mean(); return np.sqrt(np.mean((p - o)**2))/(o.max() - o.min())
sm = lambda s: pd.Series(s).rolling(7, center=True, min_periods=4).mean().values
fig = plt.figure(figsize=(jw.COL2, 7.75)); FH = fig.get_figheight(); I = lambda inch: inch/FH
L0, GAPX, WL, WR = 0.085, 0.022, 0.62, 0.27
ROW_H, TOP = 0.66, 7.18; GAPY = 0.235
out = {}
for j, dam in enumerate(EX):
    y0 = TOP - (j + 1)*ROW_H - j*GAPY
    aL = fig.add_axes([L0, I(y0), WL, I(ROW_H)]); aR = fig.add_axes([L0 + WL + GAPX, I(y0), WR, I(ROW_H)])
    d = hind(dam)                                      # hindcast 2010-2020 in MCM, centred on the days each series shares with the gauge
    o1, p1 = d.gauge.values, d["SWOT-GNN"].values; k = np.isfinite(o1) & np.isfinite(p1)
    o1, p1 = o1 - np.nanmean(o1[k]), p1 - np.nanmean(p1[k])
    t2 = T26; o2, p2 = x26_pair(dam); span = span26(dam) & rules_ok26(dam)
    o2, p2 = np.where(span, o2, np.nan), np.where(span, p2, np.nan); k2 = np.isfinite(o2) & np.isfinite(p2)
    o2, p2 = o2 - np.nanmean(o2[k2]), p2 - np.nanmean(p2[k2])
    extra = []
    if RULES:                                          # faint rule-model lines, same centring as SWOT-GNN in each period
        for m in RL:
            r1 = d[m].values if m in d else None
            if r1 is not None and np.isfinite(r1).sum() > 300:
                kk = np.isfinite(o1) & np.isfinite(r1); r1 = r1 - np.nanmean(r1[kk]) + 0.0; aL.plot(d.index, sm(r1), color=tint(RCLR[m]), lw=0.55, zorder=2); extra.append(r1)
            o3, r2 = rule26(m, dam)
            if r2 is not None: o3, r2 = np.where(span, o3, np.nan), np.where(span, r2, np.nan)
            if r2 is not None and np.isfinite(r2).sum() > 100:
                kk = np.isfinite(o3) & np.isfinite(r2); r2 = r2 - np.nanmean(r2[kk]); aR.plot(t2, sm(r2), color=tint(RCLR[m]), lw=0.55, zorder=2); extra.append(r2)
    for a_, t, o, p in ((aL, d.index, o1, p1), (aR, t2, o2, p2)):
        halo = lambda w: [pe.Stroke(linewidth=w + 1.1, foreground="white"), pe.Normal()]
        a_.plot(t, sm(o), color=jw.INK, lw=1.25, zorder=4, path_effects=halo(1.25)); a_.plot(t, sm(p), color=LINEB, lw=1.25, zorder=5, path_effects=halo(1.25))
    sw_ = swot26(dam); swp = None
    if sw_ is not None and len(sw_):          # on the gauge datum by the MEDIAN offset, so a few bad passes cannot shift the rest
        ts = sw_.index; gi = pd.Series(o2, index=t2).reindex(ts).values; kk = np.isfinite(gi)
        if kk.sum(): swp = (ts, sw_.values - np.median(sw_.values[kk] - gi[kk]))
    allv = np.r_[o1[np.isfinite(o1)], p1[np.isfinite(p1)], o2[np.isfinite(o2)], p2[np.isfinite(p2)]]; lo_, hi_ = np.percentile(allv, [0.2, 99.8])
    for e_ in extra:                                   # let the rule lines widen the range, but not their rare spikes
        if np.isfinite(e_).sum() > 50: q_ = np.nanpercentile(sm(e_), [2, 98]); lo_, hi_ = min(lo_, q_[0]), max(hi_, q_[1])
    pad = 0.08*(hi_ - lo_)
    for a_ in (aL, aR): a_.set_ylim(lo_ - pad, hi_ + pad); a_.tick_params(axis="both", labelsize=6.0, length=1.8, pad=1)
    if swp is not None:                        # in-range passes as rings, passes beyond the panel as triangles at its edge
        y0_, y1_ = lo_ - pad, hi_ + pad; v_ = swp[1]; inr = (v_ >= y0_) & (v_ <= y1_); hm, lm = v_ > y1_, v_ < y0_
        aR.scatter(swp[0][inr], v_[inr], s=6, facecolor="none", edgecolor=SWOT_RING, lw=0.6, zorder=6)
        if hm.any(): aR.scatter(swp[0][hm], np.full(hm.sum(), y1_ - 0.04*(y1_ - y0_)), s=10, marker="^", color=SWOT_RING, lw=0, zorder=7)
        if lm.any(): aR.scatter(swp[0][lm], np.full(lm.sum(), y0_ + 0.04*(y1_ - y0_)), s=10, marker="v", color=SWOT_RING, lw=0, zorder=7)
    aL.set_xlim(pd.Timestamp("2010-01-01"), pd.Timestamp("2020-12-31")); aR.set_xlim(pd.Timestamp("2023-07-01"), pd.Timestamp("2026-07-01"))
    yl = pd.date_range("2010-07-01", "2020-07-01", freq="12MS"); aL.set_xticks(pd.date_range("2010-01-01", "2020-01-01", freq="YS")); aL.set_xticklabels([])
    RT = pd.to_datetime(["2023-07-01", "2024-07-01", "2025-07-01"]); aR.set_xticks(RT); aR.set_xticklabels([])
    if j == len(EX) - 1:
        for y in range(2010, 2021): aL.text(pd.Timestamp(f"{y}-07-01"), -0.16, str(y), transform=mpl.transforms.blended_transform_factory(aL.transData, aL.transAxes), fontsize=6.0, ha="center", va="top")
        for t_, l_ in zip(RT, ["Jul 2023", "Jul 2024", "Jul 2025"]): aR.text(t_, -0.16, l_, transform=mpl.transforms.blended_transform_factory(aR.transData, aR.transAxes), fontsize=6.0, ha="center", va="top")
    aL.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=3, min_n_ticks=3))
    aL.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v/1000:g}k" if abs(v) >= 1000 else f"{v:g}"))
    aR.set_yticks([]); aR.spines["left"].set_visible(False)
    r = meta.loc[dam]; nm = K._clean_name(str(r["name"])).replace(" Reservoir", ""); st = K.ST.get(str(r.state), str(r.state))
    jw.seq_text_ax(aL, 0.0, 1.09, [(f"{nm}, {st}", jw.INK, "bold"), (f"{USE.get(str(r.use), str(r.use).lower())} · {r.cap:,.0f} MCM · {ECN.get(ECO.get(dam), '')}", jw.INK2, "normal")], fs=6.0, gap=0.012)
    (n1, e1, v1), (n2, e2, v2) = scores(o1, p1), scores(o2, p2)
    out[int(dam)] = dict(nrmse_hind=float(n1), nrmse_swot=float(n2), nse_hind=float(e1), nse_swot=float(e2), vr_hind=float(v1), vr_swot=float(v2),
                         pct=float(PCT[dam]), name=nm, eco=str(ECO.get(dam)))
    fe = lambda v: f"{v:.2f}".replace("-", "−")
    aL.text(1.0, 1.09, f"NSE {fe(e1)} ({ordn(int(round(PCT[dam])))} pct) · nRMSE {100*n1:.0f}% · VR {v1:.2f}", transform=aL.transAxes, fontsize=6.0, color="#1f66c1", ha="right", va="bottom", fontweight="bold")
    aR.text(1.0, 1.09, f"NSE {fe(e2)} · nRMSE {100*n2:.0f}% · VR {v2:.2f}", transform=aR.transAxes, fontsize=6.0, color=jw.BLUE8[7], ha="right", va="bottom", fontweight="bold")
    jw.letter(aL, "abcdefgh"[j], x=-0.075, y=1.0)
    if j == 0:
        aL.text(0.5, 1.42, "2010–2020 hindcast (no SWOT)", transform=aL.transAxes, fontsize=6.5, color=jw.INK, ha="center", va="bottom", fontweight="bold")
        aR.text(0.5, 1.42, "SWOT span (trained on), Jul 2023–Jun 2026", transform=aR.transAxes, fontsize=6.5, color=jw.INK, ha="center", va="bottom", fontweight="bold")
fig.text(0.012, 0.5, "storage anomaly (MCM)", rotation=90, fontsize=7, va="center", ha="left")
KEY = [("gauge", jw.INK, "bold"), ("SWOT-GAT", "#1f66c1", "bold")] + ([("MODROM-cal", "#a87200", "normal"), ("GDROM", "#c9501f", "normal"), ("ISTARF", "#11875c", "normal")] if RULES else []) + \
      [("○  SWOT observations (HALF-exact)", SWOT_RING, "normal"), ("· 7-day means; SWOT-GAT scores on daily values, VR = variability ratio", jw.INK2, "normal")]
jw.seq_text(fig, L0, I(7.63), KEY, fs=6.0, gap=0.012)
jw.save(fig, OUT, "e_fig_tt" + ("_rules" if RULES else ""))
json.dump(dict(examples=EX, nrmse=out), open(OUT/"num_e_fig_tt.json", "w"), indent=1); print(out)
