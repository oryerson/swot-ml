"""Summarize the expanded test sets: how many sites each adds, by tier, model set and source.

Step 5, the last step of the expanded gauge set. It only counts.

Reads (under $SWOTGAT_WORK/CONUS_storage_fork/gauges_expand)
  testsets/regime3_plus_1980_2020/sites.csv, testsets/regime1x_plus_2023_2026/sites.csv   from build_plus.py
  sizing/candidates.csv      which rule models have parameters at each dam (from size_candidates.py)

Writes
  testsets/SUMMARY.md        the tables below, also printed

Usage
  python summarize_plus.py        (no arguments)

Method
  For each of the two periods (1981-2020 and 2023-2026):
    * sites by origin (frozen_g4, candidate_additions, gauges_expand) and tier;
    * High or Medium sites per model set, split into frozen and added sites. The model sets are
      GDROM + ISTARF + calibrated MODROM + SWOT-GAT ("4-model"), GDROM + ISTARF + SWOT-GAT ("3-model"),
      ISTARF + SWOT-GAT, and SWOT-GAT alone. A rule model covers a dam if it has parameters there;
    * added sites by gauge source and tier.
  The last table gives the disk size of each gauges_expand subfolder in MB.
"""
from swotgat.paths import WORK
from pathlib import Path
import pandas as pd
F = Path(f"{WORK}/CONUS_storage_fork"); X = F/"gauges_expand"; T = X/"testsets"
C = pd.read_csv(X/"sizing/candidates.csv").drop_duplicates("GRAND_ID").set_index("GRAND_ID")
hm = lambda s: s.isin(["High", "Medium"])
rule = dict(gd=C.gdrom, ist=C.istarf, mc=C.modrom_cal)
SETS = {"4-model": lambda i: rule["gd"].reindex(i, fill_value=False) & rule["ist"].reindex(i, fill_value=False) & rule["mc"].reindex(i, fill_value=False),
        "GDROM+ISTARF (3-model)": lambda i: rule["gd"].reindex(i, fill_value=False) & rule["ist"].reindex(i, fill_value=False),
        "SWOT-GAT+ISTARF": lambda i: rule["ist"].reindex(i, fill_value=False), "SWOT-GAT alone": lambda i: pd.Series(True, index=i)}
L = ["# Expanded test sets: counts", ""]
for nm, f in (("regime3_plus_1980_2020 (1981-2020)", "regime3_plus_1980_2020"), ("regime1x_plus_2023_2026 (2023-26)", "regime1x_plus_2023_2026")):
    S = pd.read_csv(T/f/"sites.csv"); S["origin"] = S.added_by
    L += [f"## {nm}", "", "Sites by origin and tier:", "", pd.crosstab(S.origin, S.site_tier, margins=True).to_string(), ""]
    rows = []
    for sn, fn in SETS.items():
        idx = S.DAM_ID.values; m = fn(pd.Index(idx)).values; h = hm(S.site_tier).values
        fz = (S.origin == "frozen_g4").values; ca = (S.origin == "candidate_additions").values; nw = (S.origin == "gauges_expand").values
        rows.append(dict(set=sn, frozen_HM=int((m & h & fz).sum()), add_candidate_additions_HM=int((m & h & ca).sum()), add_new_HM=int((m & h & nw).sum()),
                         total_HM=int((m & h).sum()), add_new_Low=int((m & ~h & nw).sum())))
    L += ["High/Medium sites per model set (rule coverage = model has parameters at the dam):", "", pd.DataFrame(rows).to_string(index=False), ""]
    new = S[S.origin != "frozen_g4"]
    src = pd.Series("?", index=new.index)
    for c in ("ext_source", "source"):
        if c in new: src = new[c].where(new[c].notna(), src)
    L += ["Added sites by source and tier:", "", pd.crosstab(src, new.site_tier, margins=True).to_string(), ""]
sz = {p.name: round(sum(f.stat().st_size for f in p.rglob("*") if f.is_file())/1e6, 1) for p in [X/"cache", X/"discovery", X/"testsets", X/"crosswalk_plus", X/"sizing"]}
L += ["## Sizes (MB)", "", pd.Series(sz).to_string(), f"\ngauges_expand total {round(sum(f.stat().st_size for f in X.rglob('*') if f.is_file())/1e6, 1)} MB"]
(T/"SUMMARY.md").write_text("\n".join(L)); print("\n".join(L))
