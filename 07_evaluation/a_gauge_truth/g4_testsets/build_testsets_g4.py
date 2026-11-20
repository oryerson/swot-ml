"""Make corrected copies ("g4") of the reservoir test sets: one dam dropped, one dam's node list fixed.

A review of the graph found two wrong dam-to-node assignments in the frozen test sets. This script
writes a corrected copy of each test-set folder and leaves the frozen originals untouched. Both errors
were found by inspecting the graph, not by looking at model errors, so correcting them does not tune
the test sets to any model.

Reads (under $SWOTGAT_WORK/CONUS_storage_fork)
  testsets/<regime>/      for the five folders in REGIMES: regime1_2023_2025, regime1x_2023_2026,
                          regime2_2015_2020, regime2_2015_2020_v2, regime3_full_1980_2020

Writes
  eval/g4_fix/testsets/<regime>_g4/      a full copy of each folder with the two corrections
  eval/g4_fix/testsets/build_log.json    per folder: sites and tiers before and after, files changed

Usage
  python build_testsets_g4.py        (no arguments; stops if an output folder already exists)

Method
  * Dam 2158, Hopkinton Flood Control Dam, is dropped. Its gauge record is the Hopkinton Dam pool on
    the Contoocook River. The only graph node paired with it, 7310120053, is the separate Elm Brook
    Pool, and no node of graph version 2 represents the Hopkinton pool. Its rows are removed from
    every table that has a DAM_ID column.
  * Dam 294, Conconully, keeps its node 7820032943 (Conconully Reservoir, where the gauge and the
    ResOpsUS dam point are). The wrong member node 7820031672 (Spectacle Lake, 31 km away, in another
    drainage) is removed, so the dam's members are that one node. site_tier and every other column
    stay as frozen. A new column g4_note in sites.csv records the change.
  * predict_nodes.csv loses the two nodes 7310120053 and 7820031672, unless another site still uses
    them as a member.
  * Every other file is copied byte for byte. version.json gets a g4_fix block with the md5 of every
    original file.
  * Checks: every file not listed as changed has the same md5 as the original; sites.csv differs
    from the original only in the members column of dam 294, the dropped row and the new column.
"""
from swotgat.paths import WORK
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

F = Path(f"{WORK}/CONUS_storage_fork"); SRC = F/"testsets"; DST = F/"eval/g4_fix/testsets"
REGIMES = ["regime1_2023_2025", "regime1x_2023_2026", "regime2_2015_2020", "regime2_2015_2020_v2", "regime3_full_1980_2020"]
DROP = {2158}; FIX_MEMBERS = {294: ("7820032943", "7820031672")}       # dam: (keep, remove)
NOTE294 = ("G4 2026-10-19: member 7820031672 (Spectacle Lake) removed; gauge, ResOps dam point and names are Conconully "
           "Reservoir = node 7820032943. Tier left as frozen. Pending Owen's review.")
md5 = lambda p: hashlib.md5(Path(p).read_bytes()).hexdigest()
DST.mkdir(parents=True, exist_ok=True)
log = []


def dam_col(df):
    """Return the DAM_ID column of a table as integers, whether it is stored as int, float or text."""
    return df.DAM_ID.astype(str).str.replace(r"\.0$", "", regex=True).astype(int)


for r in REGIMES:
    s, d = SRC/r, DST/f"{r}_g4"
    assert not d.exists(), f"{d} exists; remove it by hand to rebuild"
    shutil.copytree(s, d)                                       # byte copies (copy2 keeps mtimes)
    changed = {}
    sites = pd.read_csv(s/"sites.csv", dtype=str)
    members_of = {m for ms in sites.loc[~dam_col(sites).isin(DROP | set(FIX_MEMBERS)), "members"].fillna("") for m in ms.split(";") if m}
    for f in sorted(p for p in s.rglob("*") if p.is_file()):
        rel = f.relative_to(s); out = d/rel
        if f.suffix == ".parquet":
            t = pd.read_parquet(f)
            if "DAM_ID" in t.columns and dam_col(t).isin(DROP).any():
                k = dam_col(t).isin(DROP); t2 = t[~k]
                t2.to_parquet(out); changed[str(rel)] = f"dropped {int(k.sum())} rows of DAM {sorted(DROP)}"
                assert t2.equals(t[~k])
        elif f.suffix == ".csv":
            t = pd.read_csv(f, dtype=str, keep_default_na=False)
            if rel == Path("sites.csv"):
                k = dam_col(t).isin(DROP); t2 = t[~k].copy(); t2["g4_note"] = ""
                for dam, (keep, rm) in FIX_MEMBERS.items():
                    m = dam_col(t2) == dam
                    if m.any():
                        assert (t2.loc[m, "node"] == keep).all() and t2.loc[m, "members"].str.contains(rm).all()
                        t2.loc[m, "members"] = keep; t2.loc[m, "g4_note"] = NOTE294
                t2.to_csv(out, index=False)
                changed[str(rel)] = f"dropped DAM {sorted(DROP)} ({int(k.sum())} row); DAM 294 members -> 7820032943; added column g4_note"
            elif rel == Path("predict_nodes.csv"):
                rm = {"7310120053", "7820031672"} - members_of          # remove only nodes no other site uses
                k = t.subbasin.isin(rm); t[~k].to_csv(out, index=False)
                changed[str(rel)] = f"removed {int(k.sum())} rows: {sorted(t.subbasin[k])}"
            elif "DAM_ID" in t.columns and dam_col(t).isin(DROP).any():
                k = dam_col(t).isin(DROP); t[~k].to_csv(out, index=False)
                changed[str(rel)] = f"dropped {int(k.sum())} rows of DAM {sorted(DROP)}"
    vj = d/"version.json"
    v = json.load(open(vj)) if vj.exists() else {}
    v["g4_fix"] = dict(built="2026-10-19", base=f"testsets/{r}", script="eval/g4_fix/build_testsets_g4.py", dropped=sorted(DROP),
                       members_fixed={str(k): v_[0] for k, v_ in FIX_MEMBERS.items()}, files_changed=changed,
                       base_md5={str(p.relative_to(s)): md5(p) for p in sorted(s.rglob("*")) if p.is_file()})
    json.dump(v, open(vj, "w"), indent=1); changed["version.json"] = "added g4_fix block (created if absent)"
    # verification: unchanged files are byte-identical; changed tables differ only as intended
    for p in sorted(x for x in s.rglob("*") if x.is_file()):
        rel = str(p.relative_to(s))
        if rel not in changed: assert md5(p) == md5(d/rel), rel
    a = pd.read_csv(s/"sites.csv", dtype=str, keep_default_na=False); b = pd.read_csv(d/"sites.csv", dtype=str, keep_default_na=False)
    keep = ~dam_col(a).isin(DROP)
    cmp = b.drop(columns="g4_note").reset_index(drop=True); ref = a[keep].reset_index(drop=True)
    diff = (cmp.fillna("") != ref.fillna(""))
    assert set(cmp.columns[diff.any()]) <= {"members"} and set(dam_col(cmp)[diff.any(axis=1)]) <= set(FIX_MEMBERS)
    tiers_a = a.site_tier.value_counts().to_dict(); tiers_b = b.site_tier.value_counts().to_dict()
    log.append(dict(regime=r, out=str(d.relative_to(F)), sites_before=len(a), sites_after=len(b), tiers_before=tiers_a, tiers_after=tiers_b,
                    dropped=[f"{x.DAM_ID} {x.DAM_NAME} ({x.site_tier})" for x in a[~keep].itertuples()],
                    members_fixed=[f"{x.DAM_ID} {x.DAM_NAME} ({x.site_tier})" for x in b[b.g4_note != ""].itertuples()] if "g4_note" in b else [],
                    files_changed=changed))
json.dump(log, open(DST/"build_log.json", "w"), indent=1)
for x in log: print(json.dumps(x, indent=1))
