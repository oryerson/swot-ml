"""Download SWOT RiverSP reach time series from the PO.DAAC Hydrocron service for a list of river reaches.

The pipeline uses this download to extend the river record past the archived RiverSP version-D tables, which end
on 2026-06-28.

Usage: python hc_pull_reach.py REACHES.txt OUT.parquet [--start YYYY-MM-DD] [--end YYYY-MM-DD]
                               [--collection NAME] [--threads N]
  REACHES.txt     one SWOT reach_id per line (reaches_graph.txt from reach_list.py)
  OUT.parquet     output table
  --start, --end  period, defaults 2026-06-01 and 2026-10-16
  --collection    Hydrocron collection name, default SWOT_L2_HR_RiverSP_D
  --threads       parallel requests, default 6

Writes
  OUT.parquet: one row per reach observation, all columns as text. The fields are those that
  merge_swot_river_conus.py reads from the archived tables (with time and time_str in place of datetime), plus
  xovr_cal_q, p_wse, cycle_id, pass_id, crid and collection_shortname. A reach whose request failed gets one row
  with the message in column `_error`.

Method
  Same request and retry logic as the lake download (04_swot_targets/a_lakes/hc_pull.py): one request per reach;
  on status 413 (response too large) the period is halved, up to 6 levels deep; status 429 and 5xx and network
  errors are retried up to 6 times with growing waits; rows marked 'no_data' are dropped.

Notes
  No login is needed. Hydrocron serves the data as it is on the day of the request, so a new download can differ
  from the one the paper used.
"""
import io, time, argparse, concurrent.futures as cf
import pandas as pd, requests
from swotgat.http import polite_get
URL = "https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1/timeseries"
FIELDS = ("reach_id,time,time_str,wse,wse_u,wse_r_u,slope,slope_u,slope_r_u,slope2,slope2_u,slope2_r_u,width,width_u,area_total,area_tot_u,"
          "area_detct,area_det_u,area_wse,layovr_val,node_dist,loc_offset,xtrk_dist,reach_q,reach_q_b,dark_frac,ice_clim_f,ice_dyn_f,partial_f,"
          "obs_frac_n,xovr_cal_q,p_wse,cycle_id,pass_id,crid,collection_shortname")
def get(rid, t0, t1, coll, depth=0):
    """Request the RiverSP observations of reach `rid` for [t0, t1] from collection `coll`.

    Returns a table of observations (text columns), an empty table if Hydrocron has no results, or a
    one-row table with the message in `_error`. `depth` counts how often the period has been halved
    after a 413 response.
    """
    p = dict(feature="Reach", feature_id=rid, start_time=f"{t0}T00:00:00Z", end_time=f"{t1}T00:00:00Z", output="csv", collection_name=coll, fields=FIELDS)
    for attempt in range(6):
        try: r = polite_get(URL, params=p, timeout=180)
        except requests.RequestException: time.sleep(5*(attempt + 1)); continue
        if r.status_code == 413 and depth < 6:
            mid = (pd.Timestamp(t0) + (pd.Timestamp(t1) - pd.Timestamp(t0))/2).date().isoformat()
            return pd.concat([get(rid, t0, mid, coll, depth + 1), get(rid, mid, t1, coll, depth + 1)])
        if r.status_code == 400 and "No results" in r.text: return pd.DataFrame()
        if r.status_code in (429, 500, 502, 503, 504): time.sleep(10*(attempt + 1)); continue
        if r.status_code != 200: return pd.DataFrame({"reach_id": [rid], "_error": [f"{r.status_code} {r.text[:200]}"]})
        d = pd.read_csv(io.StringIO(r.json()["results"]["csv"]), dtype=str)
        return d[d.time_str != "no_data"].drop(columns=[c for c in d.columns if c.endswith("_units")])
    return pd.DataFrame({"reach_id": [rid], "_error": ["retries exhausted"]})
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("reaches"); ap.add_argument("out"); ap.add_argument("--collection", default="SWOT_L2_HR_RiverSP_D")
    ap.add_argument("--start", default="2026-06-01"); ap.add_argument("--end", default="2026-10-16"); ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args(); rs = [l.strip() for l in open(a.reaches) if l.strip()]; out, t = [], time.time()
    with cf.ThreadPoolExecutor(a.threads) as ex:
        for i, d in enumerate(ex.map(lambda r: get(r, a.start, a.end, a.collection), rs)):
            out.append(d)
            if (i + 1) % 500 == 0: print(f"{i+1}/{len(rs)} reaches, {time.time()-t:.0f}s", flush=True)
    D = pd.concat(out, ignore_index=True).astype("string"); D.to_parquet(a.out, index=False)
    print(f"done: {len(rs)} reaches, {len(D)} rows, {D['_error'].notna().sum() if '_error' in D else 0} errors, {time.time()-t:.0f}s")
