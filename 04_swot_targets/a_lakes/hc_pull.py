"""Download SWOT LakeSP time series from the PO.DAAC Hydrocron service for a list of lakes.

The lakes are given by their lake_id in the SWOT Prior Lake Database (PLD). The pipeline uses this download to
build a lake record that runs past the version-C series available for 2023-07 to 2025-05.

Usage: python hc_pull.py LAKES.txt OUT.parquet [--collection NAME] [--start YYYY-MM-DD] [--end YYYY-MM-DD]
                         [--threads N]
  LAKES.txt      one lake_id per line (lakes_all.txt from lake_lists.py)
  OUT.parquet    output table
  --collection   Hydrocron collection name, default SWOT_L2_HR_LakeSP_prior_2.0
  --start, --end period, defaults 2023-07-01 and 2025-06-01
  --threads      parallel requests, default 4

Writes
  OUT.parquet: one row per lake observation with every LakeSP prior-lake attribute listed in FIELDS. All columns
  are stored as text; later scripts convert them. A lake whose request failed gets one row with the message in
  column `_error`.

Method
  One request per lake for the whole period. If Hydrocron answers 413 (response larger than 6 MB), the period is
  split in two halves and each half is requested, up to 6 levels deep. Status 429 and 5xx and network errors are
  retried up to 6 times with growing waits. "No results" gives an empty table. Rows that Hydrocron marks 'no_data'
  (a pass over the lake without an observation) are dropped, and so are the `*_units` columns.
  Requests go through swotgat.http.polite_get.

Notes
  No login is needed. Hydrocron serves the data as it is on the day of the request, so a new download can differ
  from the one the paper used.
"""
import sys, io, time, argparse, concurrent.futures as cf
import pandas as pd, requests
from swotgat.http import polite_get
URL = "https://soto.podaac.earthdatacloud.nasa.gov/hydrocron/v1/timeseries"
FIELDS = ("lake_id,reach_id,obs_id,overlap,n_overlap,time,time_tai,time_str,wse,wse_u,wse_r_u,wse_std,area_total,area_tot_u,area_detct,"
          "area_det_u,layovr_val,xtrk_dist,ds1_l,ds1_l_u,ds1_q,ds1_q_u,ds2_l,ds2_l_u,ds2_q,ds2_q_u,quality_f,dark_frac,ice_clim_f,"
          "ice_dyn_f,partial_f,xovr_cal_q,geoid_hght,solid_tide,load_tidef,load_tideg,pole_tide,dry_trop_c,wet_trop_c,iono_c,xovr_cal_c,"
          "p_res_id,p_ref_wse,p_ref_area,p_date_t0,p_ds_t0,p_storage,cycle_id,pass_id,continent_id,range_start_time,crid,"
          "collection_shortname,granuleUR,ingest_time")
def get(lake, t0, t1, coll, depth=0):
    """Request the LakeSP observations of one lake for [t0, t1] from collection `coll`.

    Returns a table of observations (text columns), an empty table if Hydrocron has no results, or a
    one-row table with the message in `_error`. `depth` counts how often the period has been halved
    after a 413 response.
    """
    p = dict(feature="PriorLake", feature_id=lake, start_time=f"{t0}T00:00:00Z", end_time=f"{t1}T00:00:00Z", output="csv",
             collection_name=coll, fields=FIELDS)
    for attempt in range(6):
        try:
            r = polite_get(URL, params=p, timeout=180)
        except requests.RequestException:
            time.sleep(5*(attempt + 1)); continue
        if r.status_code == 413 and depth < 6:
            mid = (pd.Timestamp(t0) + (pd.Timestamp(t1) - pd.Timestamp(t0))/2).date().isoformat()
            return pd.concat([get(lake, t0, mid, coll, depth + 1), get(lake, mid, t1, coll, depth + 1)])
        if r.status_code == 400 and "No results" in r.text: return pd.DataFrame()
        if r.status_code in (429, 500, 502, 503, 504): time.sleep(10*(attempt + 1)); continue
        if r.status_code != 200: return pd.DataFrame({"lake_id": [lake], "_error": [f"{r.status_code} {r.text[:200]}"]})
        csv = r.json()["results"]["csv"]
        d = pd.read_csv(io.StringIO(csv), dtype=str)          # all text: mixed-type columns (e.g. overlap) break the parquet write; convert downstream
        return d[d.time_str != "no_data"].drop(columns=[c for c in d.columns if c.endswith("_units")])
    return pd.DataFrame({"lake_id": [lake], "_error": ["retries exhausted"]})
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("lakes"); ap.add_argument("out"); ap.add_argument("--collection", default="SWOT_L2_HR_LakeSP_prior_2.0")
    ap.add_argument("--start", default="2023-07-01"); ap.add_argument("--end", default="2025-06-01"); ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(); lakes = [l.strip() for l in open(a.lakes) if l.strip()]
    out, t = [], time.time()
    with cf.ThreadPoolExecutor(a.threads) as ex:
        for i, d in enumerate(ex.map(lambda l: get(l, a.start, a.end, a.collection), lakes)):
            out.append(d)
            if (i + 1) % 50 == 0: print(f"{i+1}/{len(lakes)} lakes, {time.time()-t:.0f}s", flush=True)
    D = pd.concat(out, ignore_index=True).astype("string"); D.to_parquet(a.out, index=False)
    err = D["_error"].notna().sum() if "_error" in D else 0
    print(f"done: {len(lakes)} lakes, {len(D)} rows, {err} errors, {time.time()-t:.0f}s")
