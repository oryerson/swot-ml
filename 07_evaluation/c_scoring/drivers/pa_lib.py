"""Shared paths and helpers of the drivers-of-skill scripts in this folder.

The driver scripts (pa_*.py, fig3c_sd.py, irr_harm.py, irreg0020_g.py) build tables that describe
each scored reservoir (climate, dam, upstream regulation, gauge behavior, SWOT data quality) so
that model skill can be related to them. This module holds their input paths, the list of
reservoirs, a small graph class and a scoring helper. It reads its inputs without changing them;
outputs go to the folder out/ next to this file (eval/perf_attrib/out in the work tree), which is
created on import.

Important: several inputs belong to an earlier model configuration and an earlier graph version
  The driver analysis was set up for an earlier model configuration (a180) on graph v2.2, and
  these paths were not changed for the paper's model (see docs/KNOWN_LIMITS.md):
    OFFICIAL    score table of the earlier model. site_ids() takes the list of reservoirs from it
                (the rows with PERIOD "2010-20" and MODEL "v22a180"; 482 reservoirs) and reads
                only its id columns, never a score.
    SB, STAT    sub-basin table and static attributes of graph v2.2 (not graph v3.5).
    T3, T1X     a second copy of the 1980-2020 and 2023-2026 gauge test sets (eval/g4_fix/testsets).
    STORE_SWOT, STORE_EXT, LM, FB   training store with the SWOT targets, 1980-2025 forcing store,
                long-memory inputs and the list of lake nodes that carry a SWOT training target,
                all as built for the earlier graph versions (not the *_v35 stores).
    RUNS, PRED, OFFICIAL26, RULES_2010   prediction runs and score tables of earlier
                configurations. No script in this repository uses them.
  The quantities that reach the paper's Figure 3 (harmonic irregularity, SWOT and gauge standard
  deviations) use gauge and SWOT data only and depend on no model.

Constants
  W0, W1   2010-01-01 to 2020-12-31, the 2010-2020 scoring window
  X0, X1   2023-03-01 to 2026-09-30, the SWOT window used by the driver tables

Paths are built from WORK = $SWOTGAT_WORK (R); F = R/CONUS_storage_fork; C = R/CONUS.
"""
from swotgat.paths import WORK
import hashlib, json
import numpy as np, pandas as pd
from pathlib import Path

R = Path(f"{WORK}"); F = R/"CONUS_storage_fork"; C = R/"CONUS"
HERE = Path(__file__).resolve().parent; OUT = HERE/"out"; OUT.mkdir(exist_ok=True)
T3 = F/"eval/g4_fix/testsets/regime3_full_1980_2020_g4"; T1X = F/"eval/g4_fix/testsets/regime1x_2023_2026_g4"
OFFICIAL = F/"eval/v21_retrain/results/early_a180/compare/intermediate_sites.csv"
OFFICIAL26 = F/"eval/v21_retrain/results/early_a180/score26/sites26.csv"
RULES_2010 = F/"eval/grl_figs/intermediate/lmdft_vDspan_20261021b/figset/sites_2010_g.csv"
PRED = F/"predictions"
RUNS = {"S1": ("era5_conus_v22a180_S1_storage_20261020_032604", "step_050000"),
        "S2": ("era5_conus_v22a180_S2_storage_s1_20261020_032604", "step_050000")}
SB = C/"metadata/subbasins_conus_v22.parquet"; STAT = C/"attributes/static_conus_storage_fixgdw_v22_g4.parquet"
STORE_SWOT = C/"zarr_store_vd2026_halfexact"; STORE_EXT = C/"zarr_store_ext1980"; LM = F/"features/longmem_vd2026"
FB = C/"basin_lists/fallback/dv_SWOTAE_vd2026_fb_g.csv"
W0, W1 = pd.Timestamp("2010-01-01"), pd.Timestamp("2020-12-31")
X0, X1 = pd.Timestamp("2023-03-01"), pd.Timestamp("2026-09-30")
FROZEN = HERE/"FROZEN.json"


def site_ids():
    """Return the reservoirs of the driver tables as a sorted list of DAM_IDs (482 in the paper's run).

    They are the reservoirs that an earlier model configuration (label `v22a180`) scored in
    2010-2020. Only the id columns of that table are read, never a score.
    """
    d = pd.read_csv(OFFICIAL, usecols=["PERIOD", "MODEL", "DAM_ID"])
    return sorted(int(x) for x in d[(d.PERIOD == "2010-20") & (d.MODEL == "v22a180")].DAM_ID.unique())


def sites(path):
    """Read the sites table of a gauge test set, indexed by DAM_ID.

    path is the folder of the test set (it holds sites.csv). The added column `mem` holds each
    reservoir's member nodes as a list of strings.
    """
    s = pd.read_csv(path/"sites.csv", dtype={"node": str, "members": str, "basin": str}).set_index("DAM_ID")
    s["mem"] = [[m for m in str(v).split(";") if m and m != "nan"] for v in s.members]
    return s


class Graph:
    """The river network of graph v2.2, an earlier graph version than the paper's v3.5.

    Built from the `nextdown` column of subbasins_conus_v22.parquet. upstream(n) returns every node
    that drains to n, including n. down_path(n) returns the nodes downstream of n in order,
    without n.
    """

    def __init__(self):
        """Read the sub-basin table and build the network lookups.

        sb is the table (index = node id as string), nd maps each node to its next node downstream
        (None at an outlet), and ch maps each node to the list of nodes that drain directly into it.
        """
        sb = pd.read_parquet(SB, columns=["area_km2", "uparea_km2", "nextdown", "outlet_id"]); sb.index = sb.index.astype(str)
        self.sb = sb; self.nd = {k: (str(v) if pd.notna(v) else None) for k, v in sb.nextdown.items()}
        self.ch = {}
        for k, v in self.nd.items():
            if v is not None and v != k: self.ch.setdefault(v, []).append(k)
    def upstream(self, n):
        """Return the set of all nodes that drain to node n, including n."""
        out, st = set(), [n]
        while st:
            x = st.pop()
            if x in out: continue
            out.add(x); st.extend(self.ch.get(x, []))
        return out
    def down_path(self, n):
        """Return the list of nodes downstream of node n, in order, without n."""
        p, x, seen = [], self.nd.get(n), {n}
        while x is not None and x in self.nd and x not in seen:
            p.append(x); seen.add(x); x = self.nd.get(x)
        return p


def sha(p):
    """Return the SHA-256 hex digest of the file p."""
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check_frozen():
    """Stop if a file listed in FROZEN.json no longer has the SHA-256 recorded there.

    FROZEN.json (next to this file) maps file names to hashes. It was used to make sure that the
    variable tables were not changed after the analysis plan was fixed. Return the content of
    FROZEN.json. No script in this repository calls this function.
    """
    fz = json.load(open(FROZEN))
    for k, v in fz.items():
        assert sha(HERE/k) == v, f"FROZEN file changed: {k}"
    return fz


def centred_score(o, q, min_days):
    """Score a model series q against a gauge series o on their shared days.

    Returns None when fewer than min_days are shared or the gauge is constant. Both series have
    their own mean removed. Returns nrmse (RMSE / range of the gauge), r, amp (VR = sd ratio), n,
    the gauge's standard deviation and range, and three seasonal terms from a least-squares fit of
    mean + linear trend + annual harmonic to each series: seas_amp (ratio of harmonic amplitudes),
    and r_ns, amp_ns (correlation and sd ratio of the residuals after removing trend and
    harmonic). A constant prediction gets r = amp = seas_amp = r_ns = amp_ns = 0.
    No script in this repository calls this function.
    """
    j = pd.concat([o.rename("o"), q.rename("q")], axis=1).dropna()
    if len(j) < min_days or j.o.std() == 0: return None
    oc, qc = j.o - j.o.mean(), j.q - j.q.mean(); rng = oc.max() - oc.min(); flat = qc.std() < 1e-9*max(oc.std(), 1e-9)
    r = 0.0 if flat else float(np.corrcoef(oc, qc)[0, 1]); amp = 0.0 if flat else float(qc.std()/oc.std())
    out = dict(nrmse=float(np.sqrt(((qc - oc)**2).mean())/rng), r=r, amp=amp, n=len(j), g_sd=float(oc.std(ddof=0)), g_rng=float(rng))
    t = (j.index - j.index[0]).days.values.astype(float); Wh = 2*np.pi/365.25
    X = np.column_stack([np.ones_like(t), t/365.25, np.cos(Wh*t), np.sin(Wh*t)])
    bo = np.linalg.lstsq(X, oc.values, rcond=None)[0]; ro = oc.values - X[:, 1:] @ bo[1:]
    if flat: out.update(seas_amp=0.0, r_ns=0.0, amp_ns=0.0)
    else:
        bq = np.linalg.lstsq(X, qc.values, rcond=None)[0]; rq = qc.values - X[:, 1:] @ bq[1:]
        Ao = np.hypot(bo[2], bo[3]); out.update(seas_amp=float(np.hypot(bq[2], bq[3])/Ao) if Ao > 0 else np.nan,
                                              r_ns=float(np.corrcoef(ro, rq)[0, 1]) if ro.std() > 0 and rq.std() > 0 else 0.0,
                                              amp_ns=float(rq.std()/ro.std()) if ro.std() > 0 else np.nan)
    return out
