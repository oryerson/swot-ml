"""Record the labels of the visual crosswalk audit and compute the error rate of each tier.

Step 6c of the crosswalk. Every review sheet drawn by step6_review.py and step6b_review_batch2.py was
inspected, and each dam-to-node match was labelled correct, wrong or correct-partial. The labels are
written out in this file: the tables WRONG and PARTIAL list the dams that are not simply correct, with
the reason, and NOTES holds remarks on correct matches. Every other audited dam is labelled correct.

Reads (under $SWOTGAT_WORK/CONUS_storage_fork/crosswalk)
  crosswalk_audit.csv, crosswalk_audit_batch2.csv     the audited dams

Writes (same folder)
  crosswalk_audit_labels.csv     one row per audited dam: label, note, reviewer
  audit_summary.json             per tier: n, number wrong, error rate, Wilson 95% confidence interval;
                                 and whether the acceptance rule holds
  crosswalk_overrides.csv        one row per wrong match: link L2, new tier Reject, reason.
                                 step7_build_testsets.py applies these overrides.

Usage
  python step6c_audit_labels.py        (no arguments)

Method
  The error rate of a tier is wrong / audited, with a Wilson score interval (z = 1.96). The crosswalk
  is accepted if the upper 95% bound of the High-tier error rate is below 3%. A correct-partial match
  does not count as wrong.

Notes
  The review was a visual inspection of every sheet by an AI assistant (Claude), as the reviewer
  column of the outputs states. Spot-check the labels if you depend on them. The output tables
  crosswalk_audit_labels.csv and crosswalk_overrides.csv are shipped with the repository as frozen
  inputs (crosswalk/inputs/).
"""
from swotgat.paths import WORK
import json, numpy as np, pandas as pd
from pathlib import Path
O = Path(f"{WORK}/CONUS_storage_fork/crosswalk")
A = pd.concat([pd.read_csv(O/"crosswalk_audit.csv"), pd.read_csv(O/"crosswalk_audit_batch2.csv")], ignore_index=True)
WRONG = {1883: "Wallace: PLD polygon merges Lake Oconee (above dam) with Lake Sinclair (GRanD 1894, below); SWOT level scatters 117-127 m across two pools",
         391: "Lucky Peak: PLD polygon merges Lucky Peak with Arrowrock (GRanD 387) upstream; SWOT level scatters ~40 m"}
PARTIAL = {1263: "Twin Buttes: node is the north pool only; south pool excluded as a different SWOT level (low water separates the pools)"}
NOTES = {1125: "node correct (Lake Greeson); ResOps record is DeGray's (L1=X)", 1078: "run-of-river; polygon includes upstream river, SWOT median +5 m",
         294: "GDW outline is a different lake 35 km away; node at ResOps point and gauge (overlap r=0.976)",
         648: "GDW outline small / elevation from pre-fill DEM", 7214: "GDW elevation from pre-fill DEM", 1752: "Barkley main node after re-noding",
         1753: "Kentucky main node after re-noding", 311: "Dry Falls on Banks Lake after re-noding", 603: "SWOT +16.8 m vs ResOps: suspected SWOT height bias",
         533: "SWOT -8 m vs ResOps at a constant-level pool: suspected SWOT/datum issue"}
A["label"] = ["wrong" if d in WRONG else "correct-partial" if d in PARTIAL else "correct" for d in A.DAM_ID]
A["note"] = [WRONG.get(d) or PARTIAL.get(d) or NOTES.get(d, "") for d in A.DAM_ID]
A["reviewer"] = "Claude (visual, 2026-09-23)"
A.to_csv(O/"crosswalk_audit_labels.csv", index=False)
def wilson(k, n, z=1.96):
    """Return the Wilson score interval (low, high) of a proportion: k events in n trials.

    z is the normal quantile (1.96 for a 95% interval). The limits are clipped to 0-1.
    """
    p = k/n; d = 1 + z*z/n; c = (p + z*z/(2*n))/d; h = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return max(0.0, c-h), min(1.0, c+h)
S = {}
for t, g in A.groupby("tier"):
    k = int((g.label == "wrong").sum()); n = len(g); lo, hi = wilson(k, n)
    S[t] = dict(n=n, wrong=k, error_rate=k/n, ci95=[round(lo, 4), round(hi, 4)])
S["acceptance_rule"] = "High upper 95% CI < 3%"; S["accepted"] = bool(S["High"]["ci95"][1] < 0.03)
json.dump(S, open(O/"audit_summary.json", "w"), indent=1); print(json.dumps(S, indent=1))
ov = pd.DataFrame([dict(DAM_ID=d, link="L2", new_tier="Reject", reason=r, reviewer="Claude (visual, 2026-09-23)") for d, r in WRONG.items()])
ov.to_csv(O/"crosswalk_overrides.csv", index=False); print(ov[["DAM_ID", "new_tier"]].to_string())
