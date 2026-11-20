"""Copy the finished graph v3.5 and its model inputs into $SWOTGAT_WORK/CONUS as new *_v35 files, verified by md5.

The graph, the static attributes, the node lists and the training store are built in a staging folder
($SWOTGAT_WORK/CONUS_storage_fork/eval/v35_inputs). The model reads them from $SWOTGAT_WORK/CONUS. This script makes
that copy once all inputs are built and checked (run it after stage 04). It never replaces an existing file.

Reads
  --plan          JSON list of [source, destination] pairs; each source is a file or a folder. The paper's plan is
                  01_graph/inputs/install_plan_v35.json.template.
  --loader-check  JSON report of the check that loads the inputs with the model configuration
                  (v35_inputs/loader_check_<config stem>.json). Its "fails" list must be empty.
  v35_inputs/forcing/CHECKS.md, statics/CHECKS.md, lists/CHECKS.md   the check reports of the three input groups.
                  The script only tests that they exist; read them before running it.

Writes
  every destination in the plan
  v35_inputs/INSTALL_v35.md   one table row per destination: source and md5 (for a folder: number of files and size)

Rules
  - Every destination must lie under $SWOTGAT_WORK/CONUS/ and its name must contain "v35".
  - Nothing is copied if any source is missing or any destination already exists.
  - Files are copied, never moved. Each copy is verified by md5; for a folder, the md5 of every file is compared.

Usage
  python install_v35.py --plan plan.json --loader-check v35_inputs/loader_check_<stem>.json
"""
from swotgat.paths import WORK
import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

R = Path(f"{WORK}"); C = R/"CONUS"; V35 = R/"CONUS_storage_fork/eval/v35_inputs"
ap = argparse.ArgumentParser(); ap.add_argument("--plan", required=True); ap.add_argument("--loader-check", required=True)   # loader_check_<stem>.json
A = ap.parse_args()
plan = json.load(open(A.plan))
for f in ("forcing/CHECKS.md", "statics/CHECKS.md", "lists/CHECKS.md"):   # read by hand before this runs; must exist
    assert (V35/f).exists(), f
lc = json.load(open(A.loader_check)); assert not lc["fails"], lc["fails"]   # the loader check writes loader_check_<config stem>.json


def md5(p):
    """Return the md5 hex digest of file p, read in 4 MB blocks."""
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 22), b""): h.update(b)
    return h.hexdigest()


def tree(p):
    """Return {relative path: md5} for every file under folder p."""
    p = Path(p); return {str(f.relative_to(p)): md5(f) for f in sorted(p.rglob("*")) if f.is_file()}


# first pass: check the whole plan before copying anything
for src, dst in plan:
    src, dst = Path(src), Path(dst)
    assert src.exists(), src
    assert str(dst).startswith(str(C) + "/") and "v35" in dst.name, dst
    assert not dst.exists(), f"{dst} exists; refusing to overwrite"
lines = [f"# graph v3.5 installed into CONUS/ ({time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())})",
         "Owen 2026-10-22: \"once every check passes, write graph v3.5 and its inputs into CONUS/ as new *_v35 files (nothing overwritten)\".",
         "", "| destination | source | md5 / files |", "|---|---|---|"]
for src, dst in plan:
    src, dst = Path(src), Path(dst)
    if src.is_dir():
        shutil.copytree(src, dst)
        a, b = tree(src), tree(dst); assert a == b, f"copy mismatch {dst}"
        tag = f"{len(b)} files, {sum(os.path.getsize(dst/k) for k in b) / 1e9:.2f} GB, all md5-identical"
    else:
        shutil.copy2(src, dst); assert md5(src) == md5(dst), dst; tag = md5(dst)
    lines.append(f"| {dst} | {src} | {tag} |"); print(dst, tag, flush=True)
(V35/"INSTALL_v35.md").write_text("\n".join(lines) + "\n")
print("installed", len(plan))
