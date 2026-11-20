"""Download pre-2000 inputs of the rule-model benchmarks: statewide PDSI and GRADES discharge.

The rule-model runs that start in 1980 need a drought index (PDSI, used by GDROM's rules) and river
discharge before 2000. The sources are the same as in grades_cascade_prep.py (PDSI) and
grades_prep.py (discharge). All paths below are relative to `$SWOTGAT_WORK/GRADES/`.

Usage (one step per call):
  python grades_prep_early.py pdsi
  python grades_prep_early.py grades

Steps:
  pdsi    NOAA Climate at a Glance statewide monthly PDSI, 1979-2025, for the 48 conterminous
          states.
          Reads   pdsi_state_monthly.csv (1999-2025, from `grades_cascade_prep.py pdsi`).
          Writes  pdsi_state_monthly_1979_2025.csv (state, yyyymm, pdsi).
          Months from January 1999 on keep the values of the existing pdsi_state_monthly.csv, so
          runs for 2000-2025 do not change. Only the months before 1999 come from the new
          download. The script prints how many overlapping months agree between the two downloads
          (within 0.006) and the largest difference.
          Check printed at the end: Washington, January 1980 should be 0.54, the value in GDROM's
          cleaned training record for Ross Dam. It confirms that this is the index GDROM's rules
          were trained with.
  grades  GRADES-hydroDL v2.0 daily discharge (`Qout`) for Pfafstetter region 7, 1980-1999, at the
          same river reaches as qout_2000.parquet to qout_2025.parquet.
          Reads   reservoir_grades_map.csv, qout/qout_2000.parquet (the reach set must match,
                  else the script stops) and gdrive_listing.json (Google Drive file id of each
                  year file).
          Writes  qout/qout_<year>.parquet.
          One year file (about 770 MB) is downloaded at a time with `gdown`, reduced to the wanted
          reaches and deleted. Years already extracted are skipped. A download is tried up to
          4 times; a year that still fails is reported and skipped.

Note: the state table and the PDSI download loop are repeated here instead of imported from
grades_cascade_prep.py. That script acts on the command-line step when it is imported, so importing
it here would start its own download and overwrite pdsi_state_monthly.csv.
"""
from swotgat.paths import WORK
import sys, re, json, time, subprocess, numpy as np, pandas as pd, requests
from swotgat.http import polite_get
from pathlib import Path
G = Path(f"{WORK}/GRADES")
step = sys.argv[1]
ABBR = {"Alabama": "AL", "Arizona": "AZ", "Arkansas": "AR", "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
        "Florida": "FL", "Georgia": "GA", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS", "Kentucky": "KY",
        "Louisiana": "LA", "Maine": "ME", "Maryland": "MD", "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
        "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM",
        "New York": "NY", "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
        "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT", "Vermont": "VT",
        "Virginia": "VA", "Washington": "WA", "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY"}

if step == "pdsi":
    U = "https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/statewide/time-series/{}/pdsi/1/0/1979-2025/data.csv"
    rows = []
    for code in range(1, 49):
        txt = None
        for k in range(5):
            try:
                txt = polite_get(U.format(code), timeout=60).text; break
            except Exception: time.sleep(5 * (k + 1))
        name = re.search(r'"?([A-Za-z ]+) Palmer Drought', txt.replace("&quot;", '"')).group(1).strip()
        for line in txt.splitlines():
            m = re.match(r"^(\d{6}),(-?[\d.]+)", line)
            if m and float(m.group(2)) > -99: rows.append((ABBR[name], int(m.group(1)), float(m.group(2))))
        time.sleep(0.5)
    N = pd.DataFrame(rows, columns=["state", "yyyymm", "pdsi"])
    O = pd.read_csv(G/"pdsi_state_monthly.csv")
    j = O.merge(N, on=["state", "yyyymm"], suffixes=("_old", "_new"))
    d = (j.pdsi_old - j.pdsi_new).abs()
    print(f"download: {N.state.nunique()} states, {N.yyyymm.min()}-{N.yyyymm.max()} | overlap with existing file: {len(j)} rows, "
          f"identical {int((d < 0.006).sum())}, max |diff| {d.max():.3f}")
    M = pd.concat([N[N.yyyymm < 199901], O]).sort_values(["state", "yyyymm"]).reset_index(drop=True)
    assert not M.duplicated(["state", "yyyymm"]).any()
    wa = M[(M.state == "WA") & (M.yyyymm == 198001)].pdsi
    print(f"check WA 198001 = {wa.iloc[0] if len(wa) else 'missing'} (GDROM Ross file: 0.54)")
    M.to_csv(G/"pdsi_state_monthly_1979_2025.csv", index=False)
    print(f"wrote pdsi_state_monthly_1979_2025.csv: {len(M):,} rows, {M.yyyymm.min()}-{M.yyyymm.max()}")

elif step == "grades":
    import xarray as xr
    M = pd.read_csv(G/"reservoir_grades_map.csv", index_col=0)
    want = np.array(sorted(set(M.comid_v07.dropna().astype("int64"))))
    ref = pd.read_parquet(G/"qout"/"qout_2000.parquet")
    assert set(ref.columns.astype("int64")) == set(want), "reach set differs from the 2000-2025 extraction"
    listing = dict(json.load(open(G/"gdrive_listing.json")))
    (G/"raw").mkdir(exist_ok=True)
    for y in range(1980, 2000):
        out = G/"qout"/f"qout_{y}.parquet"
        if out.exists(): print(f"{y}: exists"); continue
        key = f"{y}/GRADES-hydroDL_V2.0_pfaf_07_{y}.nc"; nc = G/"raw"/f"pfaf_07_{y}.nc"
        if key not in listing: print("missing on Drive:", key, flush=True); continue
        for k in range(4):
            r = subprocess.run([str(G/".venv/bin/gdown"), "-q", f"https://drive.google.com/uc?id={listing[key]}", "-O", str(nc)])
            if r.returncode == 0 and nc.exists() and nc.stat().st_size > 1e8: break
            print(f"{y}: download attempt {k+1} failed (rc {r.returncode})", flush=True); time.sleep(60 * (k + 1))
        else:
            print(f"{y}: FAILED after retries", flush=True); continue
        ds = xr.open_dataset(nc); riv = ds.rivid.values.astype("int64")
        idx = np.flatnonzero(np.isin(riv, want))
        q = ds.Qout.isel(rivid=idx).load()
        df = pd.DataFrame(q.values, index=pd.to_datetime(q.time.values), columns=riv[idx]); df.index.name = "date"
        df = df[sorted(df.columns)]
        n_exp = 366 if (y % 4 == 0) else 365
        df.to_parquet(out); ds.close(); nc.unlink()
        print(f"{y}: {len(idx)} of {len(want)} reaches, {len(df)} days (expected {n_exp}), "
              f"dates {df.index.min().date()}..{df.index.max().date()}", flush=True)
    print("GRADES early extraction done")
