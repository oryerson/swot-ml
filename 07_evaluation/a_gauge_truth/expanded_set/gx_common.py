"""Shared helpers for the expanded gauge set: a cached, rate-limited download function and name matching.

discover_expand.py, harvest_expand.py and build_plus.py import this module. The expanded set adds agency
reservoir records at dams that the frozen test sets do not cover. Its working folder is
$SWOTGAT_WORK/CONUS_storage_fork/gauges_expand (X below).

Provides
  get(url, tag)    download a URL once and cache the response
  hav              great-circle distance in km between (lat, lon) points
  norm_name, sim   name similarity in [0, 1]: shared words / words of the shorter name, after generic
                   words (dam, lake, reservoir, ...) are removed
  AF               acre-feet to million cubic meters (MCM)

How get() treats the data services
  * Requests go through swotgat.http: a generic client name, one request at a time per host, and at
    least 1.1 s between two requests to the same host (5 s for HydroShare, its published crawl delay).
    The limit holds across threads.
  * Every response is cached as gauges_expand/cache/<tag>.txt.gz, so nothing is fetched twice. An
    uncompressed response already stored under the same tag (<tag>.txt, written by size_candidates.py
    in the same folder or by the 2026 harvest in gauges2026/cache) is reused, read-only.
  * Status 400 or 404 is cached as an empty response. Status 429 or 503 waits 30 s times the attempt
    number. A request is tried at most four times, with a growing pause; after that get() returns "".
  * The scripts request only the catalog entries and series of candidate dams. They do not crawl.

Notes
  The robots.txt files of some of these services disallow automated access to the data paths that
  discover_expand.py and harvest_expand.py request (size_candidates.py lists which). Check each
  service's current terms before you run the downloads again.
"""
from swotgat.paths import WORK
import gzip, re, time, threading, json, math
from pathlib import Path
from urllib.parse import urlparse
import numpy as np, pandas as pd, requests
F = Path(f"{WORK}/CONUS_storage_fork"); X = F/"gauges_expand"; CACHE = X/"cache"; CACHE.mkdir(parents=True, exist_ok=True)
OLD = [F/"gauges2026/cache", CACHE]                     # read-only reuse of earlier uncompressed responses with the same tag
from swotgat.http import PoliteSession; SES = PoliteSession()
DELAY = {"www.hydroshare.org": 5.0}; _last = {}; _locks = {}
def _lock(h):
    """Return the lock of host h, created on first use."""
    return _locks.setdefault(h, threading.Lock())
def get(url, tag, headers=None, old_ok=True, timeout=300):
    """Return the text of a URL, from the cache if it is there, otherwise by a rate-limited download.

    tag names the cache file (cache/<tag>.txt.gz; characters outside letters, digits, "_", "." and "-"
    become "_"). With old_ok, an uncompressed response stored earlier under the same tag is reused.
    A download waits for the host's lock and its minimum delay, and is tried at most four times: status
    200 is cached and returned; 400 or 404 is cached as empty and returns ""; 429 or 503 waits 30 s times
    the attempt number. Returns "" if every attempt fails.
    """
    tag = re.sub(r"[^A-Za-z0-9_.-]", "_", tag)[:180]; f = CACHE/(tag + ".txt.gz")
    if f.is_file(): return gzip.decompress(f.read_bytes()).decode("utf-8", "replace")
    if old_ok:
        for o in OLD:
            g = o/(tag + ".txt")
            if g.is_file(): return g.read_text()
    h = urlparse(url).netloc
    with _lock(h):
        for a in range(4):
            w = DELAY.get(h, 1.1) - (time.time() - _last.get(h, 0))
            if w > 0: time.sleep(w)
            try:
                r = SES.get(url, timeout=timeout, headers=headers); _last[h] = time.time()
                if r.status_code == 200: f.write_bytes(gzip.compress(r.content)); return r.text
                if r.status_code in (400, 404): f.write_bytes(gzip.compress(b"")); return ""
                if r.status_code in (429, 503): time.sleep(30*(a + 1)); continue
            except Exception:
                _last[h] = time.time()
            time.sleep(5*(a + 1))
    return ""
def hav(a, b, c, d):
    """Return the great-circle distance in km between (lat a, lon b) and (lat c, lon d).

    All coordinates are in degrees; c and d may be arrays. The Earth radius is 6371 km.
    """
    p = math.pi/180; x = np.sin((np.asarray(c) - a)*p/2)**2 + np.cos(a*p)*np.cos(np.asarray(c)*p)*np.sin((np.asarray(d) - b)*p/2)**2
    return 12742*np.arcsin(np.sqrt(x))
def norm_name(s):
    """Return a name in lower case without generic words (dam, lake, reservoir, ...) and punctuation."""
    s = str(s).lower()
    s = re.sub(r"\b(dam|lake|reservoir|res|lk|and|the|of|near|nr|at|abv|blw|tx|ca|co)\b", " ", s); s = re.sub(r"[^a-z0-9]+", " ", s)
    return " ".join(s.split())
def sim(a, b):
    """Return the similarity of two names in [0, 1]: shared words / words of the shorter name.

    Both names are first reduced with norm_name. The result is 0 if either has no word left.
    """
    a, b = set(norm_name(a).split()), set(norm_name(b).split())
    return len(a & b)/max(1, min(len(a), len(b))) if a and b else 0.0
AF = 1233.48184/1e6                                  # acre-ft -> MCM
