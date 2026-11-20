"""Write one evidence row per dam for the gauge series that is finally used in 2023-2025.

Step 5c of the crosswalk. After the tournament (step 5b) some dams use a different series than the one
tested in step 5. A dam whose series was swapped takes the overlap scores of the winning candidate, and
the replaced series id and its r are kept in two extra columns. Every other dam keeps its step-5 row.

Reads (under $SWOTGAT_WORK/CONUS_storage_fork)
  crosswalk/gauge_overlap.csv          step 5 scores
  crosswalk/gauge_tournament.csv       step 5b candidates
  reservoir_truth_2023_2025_v2.csv     which series each reservoir uses (pass == "tournament" if swapped)
  resops_targets_full.csv              node -> ResOpsUS DAM_ID

Writes
  crosswalk/gauge_final.csv            the gauge-to-dam evidence that step 7 grades (link L4)

Usage
  python step5c_gauge_final.py        (no arguments)
"""
from swotgat.paths import WORK
import pandas as pd
from pathlib import Path
F = Path(f"{WORK}/CONUS_storage_fork"); O = F/"crosswalk"
GO = pd.read_csv(O/"gauge_overlap.csv"); GO = GO[GO.role.str.startswith("truth")].drop_duplicates("DAM_ID")
TC = pd.read_csv(O/"gauge_tournament.csv")
T2 = pd.read_csv(F/"reservoir_truth_2023_2025_v2.csv", usecols=["node_orig", "sid", "pass"], dtype={"node_orig": str}).drop_duplicates("node_orig")
TGT = pd.read_csv(F/"resops_targets_full.csv", dtype={"node": str}).drop_duplicates("node").set_index("node")
T2["DAM_ID"] = T2.node_orig.map(TGT.DAM_ID)
sw = T2[T2["pass"] == "tournament"].set_index("DAM_ID").sid
rows = []
for _, r in GO.iterrows():
    if r.DAM_ID in sw.index:
        w = TC[(TC.DAM_ID == r.DAM_ID) & (TC.sid == sw[r.DAM_ID]) & (TC.role == "candidate")]
        if len(w): rows.append(w.iloc[0].to_dict() | {"role": "truth(tournament)", "replaced_sid": r.sid, "replaced_r": r.ovl_r}); continue
    rows.append(r.to_dict())
G = pd.DataFrame(rows); G.to_csv(O/"gauge_final.csv", index=False)
ok = G[G.status == "ok"]
print(f"dams {len(G)} | overlap tested {len(ok)} | r>=0.95 {(ok.ovl_r>=.95).sum()} | 0.8-0.95 {ok.ovl_r.between(.8,.95,inclusive='left').sum()} | <0.8 {(ok.ovl_r<.8).sum()} | swapped {G.role.eq('truth(tournament)').sum()}")
