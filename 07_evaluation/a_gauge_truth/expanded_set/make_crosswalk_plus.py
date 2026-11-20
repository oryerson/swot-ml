"""Record of how crosswalk_plus.py was derived from crosswalk/step23_crosswalk.py. Not run in the release.

This script wrote crosswalk_plus.py (shipped next to it) by reading step23_crosswalk.py as text and applying
the substitutions below, each required to match exactly once. It cannot be re-run here, because the text it
searches for is the original working copy of step23_crosswalk.py, not the release copy.

crosswalk_plus.py applies the same candidate search, evidence, thresholds and tier rules as the frozen
crosswalk to dams outside it: every rule-model dam with a GRanD id that is placed on a graph node
(sizing/candidates.csv) but missing from crosswalk/dam_node_crosswalk.csv. Differences:
  * the dam list is crosswalk_plus/dams_in.csv (DAM_ID, DAM_NAME, STATE, LAT, LONG, record_status) instead of
    ResOpsUS dams with 2015-2020 data;
  * the static attribute table is the graph v2.2 one of an earlier model configuration instead of the first
    graph's (lake ids are the same across graph versions);
  * ResOpsUS pool elevation is used only where a ResOpsUS file exists; otherwise the Global Dam Watch pool test
    applies, as the frozen rules already do for records without a sea-level datum;
  * outputs go to crosswalk_plus/; a new dam whose primary node is already the primary node of a dam in the
    frozen crosswalk is marked 'Collapsed' into it.
"""
from swotgat.paths import WORK
from pathlib import Path
import pandas as pd
F = Path(f"{WORK}/CONUS_storage_fork"); X = F/"gauges_expand"; O = X/"crosswalk_plus"; O.mkdir(parents=True, exist_ok=True)
ABBR = dict(AL="Alabama", AZ="Arizona", AR="Arkansas", CA="California", CO="Colorado", CT="Connecticut", DE="Delaware", FL="Florida", GA="Georgia",
            ID="Idaho", IL="Illinois", IN="Indiana", IA="Iowa", KS="Kansas", KY="Kentucky", LA="Louisiana", ME="Maine", MD="Maryland", MA="Massachusetts",
            MI="Michigan", MN="Minnesota", MS="Mississippi", MO="Missouri", MT="Montana", NE="Nebraska", NV="Nevada", NH="New Hampshire", NJ="New Jersey",
            NM="New Mexico", NY="New York", NC="North Carolina", ND="North Dakota", OH="Ohio", OK="Oklahoma", OR="Oregon", PA="Pennsylvania",
            RI="Rhode Island", SC="South Carolina", SD="South Dakota", TN="Tennessee", TX="Texas", UT="Utah", VT="Vermont", VA="Virginia",
            WA="Washington", WV="West Virginia", WI="Wisconsin", WY="Wyoming")
C = pd.read_csv(X/"sizing/candidates.csv"); XW = pd.read_csv(F/"crosswalk/dam_node_crosswalk.csv")
D = C[(C.GRAND_ID < 10000) & ~C.GRAND_ID.isin(XW.DAM_ID) & C.lat.notna()].drop_duplicates("GRAND_ID")
D = pd.DataFrame(dict(DAM_ID=D.GRAND_ID.astype(int), DAM_NAME=D.name.fillna(""), STATE=D.st.map(lambda s: ABBR.get(s, s)), LAT=D.lat, LONG=D.lon,
                      record_status=D.resops.map({True: "resops_untested", False: "agency"})))
D.to_csv(O/"dams_in.csv", index=False); print("dams", len(D), D.record_status.value_counts().to_dict())

src = (F/"crosswalk/step23_crosswalk.py").read_text()
def rep(a, b):
    """Replace text a by text b in the global src, after checking that a occurs exactly once."""
    global src
    assert src.count(a) == 1, a[:80]; src = src.replace(a, b)
rep('O = F/"crosswalk"; sys.path.insert(0, str(F))', 'O = F/"gauges_expand/crosswalk_plus"; OX = F/"crosswalk"; sys.path.insert(0, str(F))')
i0 = src.index('Q = pd.read_csv(O/"resops_record_qa.csv")'); i1 = src.index('key_chk = pd.read_csv(O/"step0_resops_damid_key.csv").set_index("DAM_ID")')
src = src[:i0] + ('dams = pd.read_csv(O/"dams_in.csv")\ndams["GRAND"] = dams.DAM_ID.astype(int)\n'
                  'key_chk = pd.read_csv(OX/"step0_resops_damid_key.csv").set_index("DAM_ID")') + src[i1 + len('key_chk = pd.read_csv(O/"step0_resops_damid_key.csv").set_index("DAM_ID")'):]
rep('st = pd.read_parquet(C/"attributes/static_conus_storage.parquet", columns=["lake_area_km2", "uparea_km2"]).reset_index()',
    'st = pd.read_parquet(C/"attributes/static_conus_storage_fixgdw_v22_g4.parquet", columns=["lake_area_km2", "uparea_km2"]).reset_index()')
rep('W = pd.read_parquet(O/"pld_wse_graph.parquet")', 'W = pd.read_parquet(OX/"pld_wse_graph.parquet")')
rep('    f = F/f"ResOpsUS/time_series_all/ResOpsUS_{dam_id}.csv"\n', '    f = F/f"ResOpsUS/time_series_all/ResOpsUS_{dam_id}.csv"\n    if not f.exists(): return (float("nan"),)*3\n')
i2 = src.index("# ---------------- invariants")
src = src[:i2] + '''# ---------------- crosswalk_plus ending: collapse onto frozen live dams, write outputs
XF = pd.read_csv(OX/"dam_node_crosswalk.csv", dtype={"node": str})
frozen_live = XF[XF.tier.isin(["High", "Medium", "Low"])].drop_duplicates("node").set_index("node").DAM_ID
k = X.node.isin(frozen_live.index) & X.tier.isin(["High", "Medium", "Low"])
X.loc[k, "collapsed_into"] = X.node[k].map(frozen_live).astype(int).astype(str) + " (frozen crosswalk)"; X.loc[k, "tier"] = "Collapsed"
live = X[X.tier.isin(["High", "Medium", "Low"])]
assert live.node.map(type).eq(str).all() and not live.node.duplicated().any() and live.node.isin(GRAPH).all()
X.to_csv(O/"dam_node_crosswalk_plus.csv", index=False)
json.dump(dict(thresholds=TH, n=len(X), tiers=X.tier.value_counts().to_dict(), source="crosswalk/step23_crosswalk.py staged by make_crosswalk_plus.py"),
          open(O/"dam_node_crosswalk_plus_meta.json", "w"), indent=1)
print("TIERS:", X.tier.value_counts().to_dict()); print(X.groupby("record_status").tier.value_counts().unstack(fill_value=0).to_string())
'''
src = src.replace('CAND.to_csv(O/"dam_node_candidates.csv", index=False)', 'CAND.to_csv(O/"dam_node_candidates_plus.csv", index=False)')
(O/"crosswalk_plus.py").write_text(src); print("wrote", O/"crosswalk_plus.py")
