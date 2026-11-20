"""Write draft captions for the four figures as Markdown, with every number read from the figures' JSON files.

Usage: mk_captions_d.py [KIT]
  KIT  name of the kit folder inside the figure-set folder; default kit_d. Run it after the four figure
       scripts.

Reads (figure-set folder = CONUS_storage_fork/eval/grl_figs/intermediate/c5_o50_20261113/ under
$SWOTGAT_WORK):
  <KIT>/num_c5d_fig1.json ... num_c5d_fig4.json  the numbers fig1_c5d.py ... fig4_c5d.py drew
  <KIT>/excl_d.py                                the reservoirs excluded from the rule-model panels
  <KIT>/cmaps_d.py                               the names of the color scales
  score/picks.json, score/picks_d.json           sizes of the pools the Figure 1 reservoirs were chosen from

Writes: captions_c5_o50_20261113d.md in the figure-set folder, and prints the same text. For any KIT other
than kit_d the file is captions_c5_o50_20261113d_ex7.md.

Method: no number is typed into the text. Counts, medians, percentiles, correlations and the table of
Figure 1 scores all come from the JSON files, so the captions cannot drift from the figures. The opening
paragraph restates the scoring rules and lists the excluded reservoirs.
"""
from swotgat.paths import WORK
import json
from pathlib import Path
B = Path(f"{WORK}/CONUS_storage_fork/eval/grl_figs/intermediate/c5_o50_20261113"); K = B/(__import__("sys").argv[1] if len(__import__("sys").argv) > 1 else "kit_d"); EXD = __import__("runpy").run_path(str(K/"excl_d.py"))["EXCL"]
import sys; sys.path.insert(0, str(K)); import cmaps_d as CB
F1 = json.load(open(K/"num_c5d_fig1.json")); F2 = json.load(open(K/"num_c5d_fig2.json")); F3 = json.load(open(K/"num_c5d_fig3.json")); F4 = json.load(open(K/"num_c5d_fig4.json"))
PD = json.load(open(B/"score/picks_d.json")); SUF = "d" if K.name == "kit_d" else "d_ex7"
XT = (f"Excluded from every set that involves the rule models (the four-model sets, the rule ensemble, Fig 2b–c, Fig 3b, Fig 3d and Fig 4): {len(EXD)} reservoirs "
      f"where the published ISTARF-CONUS parameters leave ISTARF at capacity or empty on ≥ 95% of days under the GRADES inflow (DAM_ID {', '.join(map(str, EXD))}). "
      "They stay in the SWOT-GAT-only panels (Fig 2a, Fig 3a, Fig 3c), whose scores they do not affect.")
PA = (f"Row a replaces Black Butte (ISTARF pinned): the reservoir nearest the 90th percentile of SWOT-GAT 2010–2020 NSE among the {PD['n_set_ex5']} four-model reservoirs, "
      f"from those with a gauge record on ≥ 95% of 2010–2020's days, ≥ 20 SWOT passes, in the SWOT Period four-model set, visibly seasonal (≥ 50% of 2010–2020 gauge variance in the annual harmonic) "
      f"and no pinned ISTARF run ({PD['n_pool']} reservoirs).")
PK = json.load(open(B/"score/picks.json")); f2 = lambda v: f"{v:.2f}".replace("-", "−")
S = F1["series"]; rows = sorted(S.values(), key=lambda r: r["row"])
L = [f"# Captions: SWOT-ResOps_figures_c5_o50_20261113{SUF}", "",
     "Model: SWOT-GAT = C5 (two-seed mean, step 50,000). Scoring follows overlap50: a reservoir is scored in a window when gauge and model share ≥ 50% of the window's days; both series are centred over those days; NSE = 1 − Σ(S − O)²/ΣO² on the centred series; VR = model sd ÷ gauge sd; nRMSE = RMSE ÷ gauge range. High- and Medium-tier reservoirs only.", "", XT, "",
     "## Figure 1. Daily storage anomalies at eight reservoirs: gauge, SWOT-GAT and the three rule models",
     "",
     f"Rows a–d ({', '.join(r['name'] for r in rows[:4])}): the 2010–2020 hindcast (left) and the SWOT Period, Jul 2023–Jun 2026 (right; each reservoir's SWOT span, the days every model has a value). Rows e–h ({', '.join(r['name'] for r in rows[4:])}): the full 1981–2020 hindcast (left) and the SWOT Period (right). Lines are 7-day running means for display; scores are on daily values. Each series is centred on its mean over the days it shares with the gauge. Dark blue rings: SWOT observations (HALF-exact vD storage, member lakes summed on all-member days), placed on the gauge's datum.",
     f"Scores above each panel are SWOT-GAT's. pct = percentile of SWOT-GAT's hindcast NSE: rows a–d among the {F1['n_pct_2010']} reservoirs all four models score in 2010–2020; rows e–h among the {F1['n_pct_full']} reservoirs SWOT-GAT is scored at over 1981–2020.",
     f"Rows e–h were chosen by rule from reservoirs with a gauge record on ≥ 95% of 1981–2020's days, all three rule models scored over 1981–2020, in the SWOT Period four-model set and with ≥ 20 SWOT passes ({PK['n_pool']} reservoirs): the one nearest the 90th, 60th, 35th and 10th percentile of SWOT-GAT 1981–2020 NSE.",
     PA, "", "| row | reservoir | window | NSE | pct | nRMSE | VR | SWOT Period NSE | VR |", "|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    L.append(f"| {r['row']} | {r['name']}, {r['state']} | {r['window'].replace('-', '–')} | {f2(r['nse_hind'])} | {r['pct']:.0f} | {100*r['nrmse_hind']:.0f}% | {r['vr_hind']:.2f} | {f2(r['nse_swot'])} | {r['vr_swot']:.2f} |")
a = F2["a"]; b = F2["8120"]; c = F2["2023-25"]
L += ["", "## Figure 2. SWOT-GAT and rule-model NSE over the full hindcast and the SWOT Period", "",
      f"(a) SWOT-GAT NSE over 1981–2020 at {a['n']} reservoirs (median {f2(a['median_nse'])}); dot size = capacity. Colours ({CB.NSE_LABEL}) span the 10th–90th percentile ({f2(a['p10'])} to {f2(a['p90'])}); the bar's end triangles hold the {100*a['share_below_p10']:.0f}% below and {100*a['share_above_p90']:.0f}% above."
      + (f" Bonny (CO) has no row in sites_v5_g.csv; its location and capacity are GRanD's (gauges_expand candidates.csv)." if a.get("coords_from_grand") else ""),
      f"(b) 1981–2020 and (c) SWOT Period: cumulative distributions of NSE by main purpose. Solid: SWOT-GAT. Dashed: the rule-model ensemble, the daily mean of GDROM, ISTARF and MODROM-cal storage on days all three have a value, scored like any model. Reservoirs where both are scored: {b['n']} in 1981–2020 (the four-model set) and {c['n']} in the SWOT Period. NSE below −1 is drawn at −1. The tables give n and the median NSE; reservoirs with another main purpose ({b['n_other']} and {c['n_other']}) are in the all-reservoir row only.",
      f"All reservoirs, median NSE SWOT-GAT / ensemble: 1981–2020 {f2(b['all']['GAT'])} / {f2(b['all']['ENS'])} (NSE < 0 at {100*b['all']['GAT_lt0']:.0f}% / {100*b['all']['ENS_lt0']:.0f}%; ensemble higher at {100*b['all']['ENS_better']:.0f}% of reservoirs); SWOT Period {f2(c['all']['GAT'])} / {f2(c['all']['ENS'])} ({100*c['all']['GAT_lt0']:.0f}% / {100*c['all']['ENS_lt0']:.0f}% below 0; ensemble higher at {100*c['all']['ENS_better']:.0f}%).", ""]
a3, b3, c3, d3 = F3["a"], F3["b"], F3["c"], F3["d"]
L += ["## Figure 3. Variability ratio of SWOT-GAT's 1981–2020 hindcast", "",
      f"(a) SWOT-GAT variability ratio over 1981–2020 at {a3['n']} reservoirs (median {a3['median']:.2f}; below 1 at {100*a3['share_below_1']:.0f}%); colours ({CB.VR_LABEL}) on a log scale over the 10th–90th percentile ({a3['p10']:.2f}–{a3['p90']:.2f}).",
      f"(b) 1981–2020 NSE of the four models by thirds of storage irregularity (sd of the gauge around its fitted annual + semiannual harmonic ÷ the harmonic's sd, water years 1981–2020; tertile edges {b3['edges'][1]:.2f} and {b3['edges'][2]:.2f} over every reservoir with a value), on the {b3['n_set']} reservoirs all four models score over 1981–2020 ({', '.join(str(b3[k]['n']) for k in '012')} per third). Boxes: 25th–75th percentile, line median, whiskers 5th–95th, drawn at the axis limit beyond it.",
      f"(c) SWOT-GAT 1981–2020 variability ratio against the training variability ratio sd(SWOT storage, SWOT Period) ÷ sd(gauge, 1981–2020), the same gauge sd period as the VR denominator, so on the dashed 1:1 line the hindcast varies as much as SWOT observed; below it, less. Points coloured by SWOT-GAT 1981–2020 NSE on Fig 2a's scale. Spearman ρ = {c3['rho'][0]:+.2f} [{c3['rho'][1]:+.2f}, {c3['rho'][2]:+.2f}] (block bootstrap), n = {c3['n']}; partial ρ given irregularity {c3['partial_irr_harm']:+.2f}; {100*c3['share_below_1to1']:.0f}% of reservoirs below 1:1.",
      f"(d) Variability ratio distributions for SWOT-GAT and MODROM-cal: solid 1981–2020 (n = {d3['8120']['n']}), dashed SWOT Period (n = {d3['2023-25']['n']}). Medians SWOT-GAT {d3['8120']['v10pm']:.2f} / {d3['2023-25']['v10pm']:.2f}; MODROM-cal {d3['8120']['MODROM-cal']:.2f} / {d3['2023-25']['MODROM-cal']:.2f}.", "",
      "## Figure 4. 2010–2020 hindcast by main purpose", "",
      f"(a) NSE and (b) variability ratio of the four models by main purpose, on the {next(v for k, v in F4['c'].items() if k.startswith('All'))['n']} reservoirs all four score in 2010–2020; (c) share of reservoirs where each model has the highest NSE.", ""]
(B/f"captions_c5_o50_20261113{SUF}.md").write_text("\n".join(L)); print("\n".join(L))
