"""Record of how stage/gauge_stat_screen.py was derived from gauge_stat_screen.py. Not run in the release.

gauge_stat_screen.py runs its whole audit, and rewrites its output files, whenever it is imported. This script
wrote a copy (shipped as stage/gauge_stat_screen.py, the one build_plus.py imports) in which everything after
the screen() function sits under `if __name__ == "__main__":` and the output folder can be set with the
environment variable GSS_OUT, so that importing `screen` writes nothing. It cannot be re-run here, because
the text it searches for is the original working copy of gauge_stat_screen.py, not the release copy.

It also checked that the audit outputs on disk equal a fresh recomputation from the frozen inputs, and that
the printed audit summary equals every earlier log of it.
"""
from swotgat.paths import WORK
import os, re, subprocess, sys, tempfile, hashlib
from pathlib import Path
import pandas as pd
F = Path(f"{WORK}/CONUS_storage_fork"); X = F/"gauges_expand"; S = X/"stage"; S.mkdir(exist_ok=True)
src = (F/"eval/gauge_stat_screen.py").read_text()
a = 'OUT = F/"eval/swotqc/audit"'; assert src.count(a) == 1
src = src.replace(a, 'import os\nOUT = Path(os.environ.get("GSS_OUT", str(F/"eval/swotqc/audit")))')
k = src.index("\nrows, summ = [], []\n"); body = src[k + 1:]
src = src[:k + 1] + 'if __name__ == "__main__":\n' + "".join("    " + l if l.strip() else l for l in body.splitlines(True))
src = '"""[gauges_expand staged copy of eval/gauge_stat_screen.py: audit only under __main__, output folder GSS_OUT]"""\n' + src
(S/"gauge_stat_screen.py").write_text(src)
sys.path.insert(0, str(S)); import gauge_stat_screen as G; assert hasattr(G, "screen")          # import must not write
tmp = Path(tempfile.mkdtemp(prefix="gss_", dir=str(X/"stage")))
r = subprocess.run([sys.executable, "-W", "ignore", str(S/"gauge_stat_screen.py")], env={**os.environ, "GSS_OUT": str(tmp)}, capture_output=True, text=True)
print(r.stdout[-3000:], r.stderr[-2000:])
md5 = lambda p: hashlib.md5(open(p, "rb").read()).hexdigest()
for f in ("gauge_stat_flags.parquet", "gauge_stat_summary.csv"):
    a, b = F/"eval/swotqc/audit"/f, tmp/f
    da = pd.read_parquet(a) if f.endswith("parquet") else pd.read_csv(a); db = pd.read_parquet(b) if f.endswith("parquet") else pd.read_csv(b)
    print(f, "| md5 on disk", md5(a), "| md5 recomputed", md5(b), "| content equal:", da.equals(db), da.shape, db.shape)
# printed summary vs older logs
block = r.stdout.strip()
logs = [p for p in list((F/"eval").glob("*.log")) + list((F/"eval/overnight").glob("*.log")) + list((F/"eval/swotqc").rglob("*.log"))]
hits = []
for p in logs:
    try: t = p.read_text(errors="replace")
    except Exception: continue
    if "series per set:" in t and "quantized series:" in t:
        hits.append((p, block in t))                       # the whole printed audit (flag table + series lines) appears verbatim
for p, ok in hits: print("older log", p.relative_to(F), "| summary lines identical:", ok)
