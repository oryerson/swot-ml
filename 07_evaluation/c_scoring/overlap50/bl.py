"""Two simple baselines that the scorers can add as extra models, and the switch that turns them on.

This file exists in two identical copies (overlap50/ and full_record/),
because each scorer imports the bl.py that sits next to it.

Usage
  The scorers import this module and read the environment variable BASE:
    BASE=0 or unset   no baseline is added; the scorers' tables hold only the models.
    BASE=1            the scorers add the two baselines below and score them like any other model.
  Importing the module only defines the functions and loads normal_year() (see below). Running the
  file as a script builds nothing. The fit table of the first baseline must be built once, before
  the first BASE=1 run, by calling build_fit() from the folder that holds this file, with the
  repository's src/ folder on PYTHONPATH (for swotgat.paths):
    python -c "import bl; bl.build_fit()"

Baselines
  SWOT-cycle    Baseline (a). Uses no gauge data. A constant plus an annual and a semiannual harmonic
                (period 365.25 days) is fitted by ordinary least squares to the reservoir's SWOT
                storage-change series from 2023-01-01 to 2026-09-30. The series comes from the SWOT
                LakeSP version D lake table after the HALF outlier filter and its coverage rule.
                Per lake and day the value is dv_SWOTAE, else dv_3DLAKES, else dv_staticPLD (storage
                change from the SWOT area-elevation curve, from the 3D-LAKES curve, or from level
                change times the static lake area of the SWOT Prior Lake Database, PLD). Values are
                summed over the reservoir's member lakes on days when every member has one. This is
                the series that 07_evaluation/d_swot_vs_gauge/swot_vs_gauge.py compares with gauges.
                A fit is usable when the series has at least 20 days that span at least 300 days.
                The fitted function is evaluated on every day of every scored window, so the
                baseline repeats the same cycle every year.
  normal_year   Baseline (b). Uses the gauge. It is the reservoir's mean day-of-year anomaly cycle
                (anomalies from each year's median, 31-day circular smoothing), computed from its
                gauge record outside the scored window; at least 3 reference years are required.
                The function normal_year() is not written in this file. It is cut from the source
                text of an earlier full-record scorer (SRC_FR; in this repository
                normal_year_source/score_fullrecord.py) and executed, so it is that scorer's
                function exactly. The gauge record is the 1980-2020 test set, restricted to days
                whose `use` flag is set. The full-record scorer (score_fullrecord.py) scores
                normal_year itself; this module adds it to the 2010-2020 and SWOT Period scorers.

Reads (paths relative to $SWOTGAT_WORK/CONUS_storage_fork)
  eval/half_vd_review/lakesp_vD_halfexact_graph.parquet     filtered LakeSP version D (build_fit)
  testsets/regime3_full_1980_2020/sites.csv, truth_clean.parquet   the 1980-2020 gauge test set
  testsets/regime1x_2023_2026/sites.csv                     the 2023-2026 gauge test set (members)
  eval/c5_paper/fig_c5/score_fullrecord.py                  source text of normal_year()
  eval/baselines_c5/score/swot_fit.csv                      the fit table, when a cycle is requested

Writes (build_fit only)
  eval/baselines_c5/score/swot_fit.csv: one row per dam with the number and span of SWOT days,
  `usable` and the reason when a fit is not usable, the five coefficients (c, a1, b1, a2, b2) and
  fit diagnostics (rank, condition number, R2, standard deviations, harmonic amplitudes).
"""
from swotgat.paths import WORK
import os, numpy as np, pandas as pd
from pathlib import Path
F = Path(f"{WORK}/CONUS_storage_fork"); B = F/"eval/baselines_c5"
BASE = os.environ.get("BASE", "0") == "1"
T3 = F/"testsets/regime3_full_1980_2020"; X1 = F/"testsets/regime1x_2023_2026"
HALF = F/"eval/half_vd_review/lakesp_vD_halfexact_graph.parquet"
FIT0, FIT1, MIN_DAYS, MIN_SPAN = pd.Timestamp("2023-01-01"), pd.Timestamp("2026-09-30"), 20, 300
OMEGA, T0 = 2*np.pi/365.25, pd.Timestamp("2000-01-01")
FIT_CSV = B/"score/swot_fit.csv"
A_NAME, B_NAME = "SWOT-cycle", "normal_year"
COEF = ["c", "a1", "b1", "a2", "b2"]
SRC_FR = F/"eval/c5_paper/fig_c5/score_fullrecord.py"


def _verbatim(name, src=SRC_FR):
    """Return a top-level function cut from a scorer's source text, together with that text.

    The text between `def <name>(` and the next top-level `def` is executed unchanged, so the result
    is the scorer's own function by construction.
    """
    s = Path(src).read_text(); i = s.index(f"\ndef {name}(") + 1; j = s.index("\ndef ", i)
    ns = {"np": np, "pd": pd}; exec(compile(s[i:j], f"{src.name}:{name}", "exec"), ns); return ns[name], s[i:j]


normal_year, NORMAL_YEAR_SRC = _verbatim("normal_year")


def members():
    """Map each DAM_ID to its member lakes.

    The list of the 1980-2020 test set is used if the dam is there, otherwise the list of the
    2023-2026 test set. In the paper's run the two lists are identical at all 456 shared dams.
    Return a dict DAM_ID -> list of lake ids (strings).
    """
    out = {}
    for f in (X1, T3):
        for r in pd.read_csv(f/"sites.csv", dtype={"members": str}).itertuples(): out[int(r.DAM_ID)] = str(r.members).split(";")
    return out


def design(t):
    """Return the design matrix of the cycle for the times t, in days since T0.

    The columns are a constant, the annual cosine and sine and the semiannual cosine and sine, in
    the order of COEF.
    """
    t = np.asarray(t, float)
    return np.column_stack([np.ones_like(t), np.cos(OMEGA*t), np.sin(OMEGA*t), np.cos(2*OMEGA*t), np.sin(2*OMEGA*t)])


def days(dates):
    """Return the dates as days since T0 (2000-01-01), the time axis of the fit."""
    return np.asarray((pd.DatetimeIndex(dates) - T0).days, float)


def build_fit():
    """Fit the SWOT-cycle baseline for every dam of the two gauge test sets and write the fit table.

    Writes eval/baselines_c5/score/swot_fit.csv and returns it as a DataFrame. Scores nothing.
    """
    HE = pd.read_parquet(HALF, columns=["lake_id", "date", "dv_SWOTAE", "dv_3DLAKES", "dv_staticPLD"])
    HE["lake_id"] = HE.lake_id.astype(str); HE["date"] = pd.to_datetime(HE.date)
    assert not HE.duplicated(["lake_id", "date"]).any(), "duplicate lake-days in the HALF-exact record"
    HE["v"] = HE.dv_SWOTAE.fillna(HE.dv_3DLAKES).fillna(HE.dv_staticPLD)
    HE["src"] = np.select([HE.dv_SWOTAE.notna(), HE.dv_3DLAKES.notna()], ["SWOTAE", "3DLAKES"], "staticPLD")
    HE = HE[HE.v.notna()]; HEV = {k: g.set_index("date")[["v", "src"]] for k, g in HE.groupby("lake_id")}
    rows = []
    for dam, mem in sorted(members().items()):
        row = dict(DAM_ID=dam, n_members=len(mem), members=";".join(mem), n_days=0, span_days=0)
        miss = [m for m in mem if m not in HEV]
        if miss:
            rows.append({**row, "usable": False, "why": f"no HALF-exact series for {len(miss)} of {len(mem)} member lake(s)"}); continue
        v = pd.concat([HEV[m].v for m in mem], axis=1).dropna(); s = pd.concat([HEV[m].src for m in mem], axis=1).reindex(v.index)
        y = v.sum(axis=1); keep = (y.index >= FIT0) & (y.index <= FIT1); y, s = y[keep], s[keep]
        n = len(y); span = int((y.index.max() - y.index.min()).days) if n else 0
        row.update(n_days=n, span_days=span, first=str(y.index.min().date()) if n else "", last=str(y.index.max().date()) if n else "",
                   share_all_swotae=float((s == "SWOTAE").all(axis=1).mean()) if n else np.nan)
        if n < MIN_DAYS or span < MIN_SPAN:
            rows.append({**row, "usable": False, "why": f"{n} days spanning {span} days (needs >= {MIN_DAYS} and >= {MIN_SPAN})"}); continue
        X = design(days(y.index)); b, _, rank, sv = np.linalg.lstsq(X, y.values.astype(float), rcond=None); fit = X @ b
        yc = y.values - y.values.mean()
        row.update(usable=True, why="", **dict(zip(COEF, map(float, b))), rank=int(rank), cond=float(sv[0]/sv[-1]),
                   r2_fit=float(1 - ((y.values - fit)**2).sum()/(yc**2).sum()) if yc.std() > 0 else np.nan,
                   sd_swot=float(y.std()), sd_cycle=float(fit.std()), amp_annual=float(np.hypot(b[1], b[2])), amp_semi=float(np.hypot(b[3], b[4])),
                   n_years_touched=int(y.index.year.nunique()))
        rows.append(row)
    D = pd.DataFrame(rows); FIT_CSV.parent.mkdir(parents=True, exist_ok=True); D.to_csv(FIT_CSV, index=False)
    print(f"bl.build_fit: {len(D)} dams, usable {int(D.usable.sum())}; reasons: {D[~D.usable].why.str.replace(r'[0-9]+', '#', regex=True).value_counts().to_dict()}", flush=True)
    return D


_FIT, _G = None, None


def fits():
    """Return the fit table indexed by DAM_ID, read once and kept in memory."""
    global _FIT
    if _FIT is None: _FIT = pd.read_csv(FIT_CSV).set_index("DAM_ID")
    return _FIT


def swot_cycle(dam, dates):
    """Return the SWOT-cycle baseline on the given dates, or None when the dam has no usable fit."""
    f = fits(); dam = int(dam)
    if dam not in f.index or not bool(f.at[dam, "usable"]): return None
    dates = pd.DatetimeIndex(dates); return pd.Series(design(days(dates)) @ f.loc[dam, COEF].values.astype(float), index=dates)


def swot_cycles(dams, dates):
    """Return the SWOT-cycle baseline of many dams as {dam: series}.

    Dams without a usable fit are left out.
    """
    out = {}
    for d in dams:
        s = swot_cycle(d, dates)
        if s is not None: out[int(d)] = s
    return out


def gauge_full():
    """Map each dam to its gauge record from the 1980-2020 test set (days with the `use` flag set).

    Built the same way as score_fullrecord.py builds its gauge series `oa`. Read once and kept.
    Return a dict DAM_ID -> daily storage Series (date index).
    """
    global _G
    if _G is None:
        TR = pd.read_parquet(T3/"truth_clean.parquet"); TR["date"] = pd.to_datetime(TR.date); TR = TR[TR.use.fillna(False).astype(bool)]
        _G = {int(k): g.set_index("date").storage_mcm.sort_index() for k, g in TR.groupby("DAM_ID")}
    return _G


def gauge_cycle(dam, w0, w1, dates):
    """Return the normal_year baseline on the given dates, from the gauge record outside [w0, w1].

    Returns None when the dam has no record, fewer than 3 reference years, or a missing value.
    """
    oa = gauge_full().get(int(dam))
    if oa is None or oa.empty: return None
    ny = normal_year(oa, w0, w1, pd.DatetimeIndex(dates))
    return None if ny is None or ny.isna().any() else ny


def x26(dam, a0, b0, T):
    """Return both baselines on the SWOT Period dates T, as {model name: series}.

    [a0, b0] is the reservoir's scored span, the window that normal_year leaves out. A baseline
    that is not available for the dam is left out. The scorer restricts the series to the days on
    which it scores SWOT-GAT.
    """
    out = {}
    a = swot_cycle(dam, T)
    if a is not None: out[A_NAME] = a
    b = gauge_cycle(dam, a0, b0, T)
    if b is not None: out[B_NAME] = b
    return out
