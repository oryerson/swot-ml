"""Download the 2026 inputs of the rule-model benchmarks: GRADES discharge and statewide PDSI.

These extend the rule-model inputs through the SWOT Period. The sources and the extraction are the
same as in grades_prep.py (discharge) and grades_cascade_prep.py (PDSI). All paths below are
relative to `$SWOTGAT_WORK/GRADES/`.

Usage (one step per call):
  python grades_prep_2026.py grades
  python grades_prep_2026.py pdsi

Steps:
  grades  GRADES-hydroDL v2.0 daily discharge (`Qout`) for Pfafstetter region 7, 2026-01-01 to
          2026-06-12, at the MERIT-Basins v0.7 river reach of every reservoir node.
          Reads   reservoir_grades_map.csv, gdrive_listing.json (Google Drive file ids),
                  qout/qout_2025.parquet.
          Writes  qout/qout_2026.parquet.
          The file is downloaded with `gdown`, reduced to the wanted reaches and deleted. The
          script prints the date range, the time steps, the share of missing values, and a
          continuity check: the ratio of mean discharge in the first week of 2026 to the last week
          of 2025 at the same reaches (median and interquartile range).
  pdsi    NOAA Climate at a Glance statewide monthly PDSI, 2025-2026, for the 48 conterminous
          states.
          Reads   pdsi_state_monthly_1979_2025.csv (from `grades_prep_early.py pdsi`).
          Writes  pdsi_state_monthly_1979_2026.csv: the 1979-2025 table with the 2026 months
                  appended.
          The 2025 values of the existing table are kept. The script prints how many 2025 months
          NOAA has revised since (difference above 0.006) and by how much, and the last 2026 month
          available per state.

Notes:
  - Neither step overwrites: each stops if its output file already exists.
  - The state table and the PDSI download loop are repeated from grades_cascade_prep.py.
"""
from swotgat.paths import WORK
import sys, re, json, time, subprocess, numpy as np, pandas as pd, xarray as xr, requests
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

if step == "grades":
    out = G/"qout"/"qout_2026.parquet"
    assert not out.exists(), f"{out} exists; not overwriting"
    M = pd.read_csv(G/"reservoir_grades_map.csv", index_col=0)
    want = np.array(sorted(set(M.comid_v07.dropna().astype("int64"))))
    listing = dict(json.load(open(G/"gdrive_listing.json")))
    key = "2026/GRADES-hydroDL_V2.0_pfaf_07_20260101_20260612.nc"
    D = G/"raw"/"dl2026"; D.mkdir(parents=True, exist_ok=True); nc = D/"pfaf_07_20260101_20260612.nc"
    for k in range(3):
        r = subprocess.run([str(G/".venv/bin/gdown"), "-q", f"https://drive.google.com/uc?id={listing[key]}", "-O", str(nc)])
        if r.returncode == 0 and nc.exists() and nc.stat().st_size > 5e7: break
        time.sleep(30)
    print(f"downloaded {nc.name}: {nc.stat().st_size/1e6:.1f} MB", flush=True)
    ds = xr.open_dataset(nc); print(ds, flush=True)
    riv = ds.rivid.values.astype("int64"); idx = np.flatnonzero(np.isin(riv, want))
    q = ds.Qout.isel(rivid=idx).load()
    df = pd.DataFrame(q.values, index=pd.to_datetime(q.time.values), columns=riv[idx]); df.index.name = "date"
    t = df.index
    print(f"2026: {len(idx)} of {len(want)} reaches, {len(df)} days {t.min().date()}..{t.max().date()}, "
          f"daily steps unique {sorted(set(np.diff(t.values).astype('timedelta64[h]').astype(int)))} h, NaN share {df.isna().mean().mean():.2e}, "
          f"dtype {df.dtypes.iloc[0]}", flush=True)
    # continuity with 2025: last week of 2025 vs first week of 2026 at the same reaches
    q25 = pd.read_parquet(G/"qout"/"qout_2025.parquet"); q25.columns = q25.columns.astype("int64")
    a, b = q25.iloc[-7:][df.columns].mean(), df.iloc[:7].mean()
    ok = (a > 1) & (b > 0)
    print(f"continuity Dec-25 last week -> Jan-26 first week: median ratio {np.median(b[ok]/a[ok]):.3f}, "
          f"IQR {np.percentile(b[ok]/a[ok],25):.3f}-{np.percentile(b[ok]/a[ok],75):.3f} (n={int(ok.sum())})", flush=True)
    df.to_parquet(out); ds.close(); nc.unlink(); D.rmdir()
    print(f"wrote {out}; deleted the .nc", flush=True)

elif step == "pdsi":
    out = G/"pdsi_state_monthly_1979_2026.csv"
    assert not out.exists(), f"{out} exists; not overwriting"
    U = "https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/statewide/time-series/{}/pdsi/1/0/2025-2026/data.csv"
    rows = []
    for code in range(1, 49):
        for k in range(4):
            try:
                txt = polite_get(U.format(code), timeout=60).text; break
            except Exception: time.sleep(5 * (k + 1))
        name = re.search(r'"?([A-Za-z ]+) Palmer Drought', txt.replace("&quot;", '"')).group(1).strip()
        for line in txt.splitlines():
            m = re.match(r"^(\d{6}),(-?[\d.]+)", line)
            if m and float(m.group(2)) > -99: rows.append((ABBR[name], int(m.group(1)), float(m.group(2))))
        time.sleep(0.5)
    N = pd.DataFrame(rows, columns=["state", "yyyymm", "pdsi"])
    O = pd.read_csv(G/"pdsi_state_monthly_1979_2025.csv")
    m = O[O.yyyymm >= 202501].merge(N, on=["state", "yyyymm"], suffixes=("_old", "_new"))
    d = (m.pdsi_new - m.pdsi_old).abs()
    print(f"states {N.state.nunique()} | NOAA months {N.yyyymm.min()}-{N.yyyymm.max()} | 2025 overlap {len(m)} months, "
          f"revised {int((d > 0.006).sum())}, max |diff| {d.max():.2f}, median |diff| of revised {d[d > 0.006].median() if (d > 0.006).any() else 0:.2f}")
    print("last 2026 month per state:", N[N.yyyymm >= 202601].groupby("state").yyyymm.max().value_counts().to_dict())
    P = pd.concat([O, N[N.yyyymm >= 202601]], ignore_index=True).sort_values(["state", "yyyymm"])
    assert not P.duplicated(["state", "yyyymm"]).any()
    P.to_csv(out, index=False)
    print(f"wrote {out.name}: {len(P)} rows, {P.yyyymm.min()}-{P.yyyymm.max()}")
