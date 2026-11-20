"""Compare smoke-test outputs with the reference values of the paper's results.

    compare.py expected_md5.tsv            # check the work tree against the table
    compare.py expected_md5.tsv --write    # (authors) rebuild the table from a reference tree in $SWOTGAT_REFERENCE

``expected_md5.tsv`` lists, for each output, its path under ``$SWOTGAT_WORK``,
the md5 of the file the paper used, and where the smoke test writes its copy.
A file passes if the md5 matches. A table that differs in bytes still passes
if every text cell is equal and every number agrees to 1e-9 (relative), which
is reported as "equal values". The file to compare values with is taken from
``$SWOTGAT_REFERENCE`` if that is set, else from ``tests/smoke/expected/``
(one file is kept there: ``num_c5d_fig1.json``, whose last digits vary
between runs).
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path

WORK = Path(os.environ["SWOTGAT_WORK"])
EXPECTED = Path(__file__).resolve().parent / "expected"
TOL = 1e-9


def md5(path: Path) -> str:
    """Return the md5 sum of a file as a hexadecimal string."""
    return hashlib.md5(path.read_bytes()).hexdigest()


def _same_value(a, b) -> bool:
    if a == b:
        return True
    try:
        x, y = float(a), float(b)
    except (TypeError, ValueError):
        return False
    if math.isnan(x) and math.isnan(y):
        return True
    return abs(x - y) <= TOL * max(1.0, abs(x), abs(y))


def _same_json(a, b) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same_json(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same_json(x, y) for x, y in zip(a, b))
    return _same_value(a, b)


def same_values(got: Path, ref: Path) -> bool:
    """Return True if two CSV or JSON files hold equal text and numbers equal to 1e-9."""
    if got.suffix == ".json":
        return _same_json(json.loads(got.read_text()), json.loads(ref.read_text()))
    if got.suffix == ".csv":
        with open(got, newline="") as f1, open(ref, newline="") as f2:
            r1, r2 = list(csv.reader(f1)), list(csv.reader(f2))
        return len(r1) == len(r2) and all(len(a) == len(b) and all(_same_value(x, y) for x, y in zip(a, b))
                                          for a, b in zip(r1, r2))
    return False


def main() -> int:
    """Compare the work tree with the reference table, or rebuild the table with ``--write``."""
    table = Path(sys.argv[1])
    rows = [r for r in csv.DictReader(open(table, newline=""), delimiter="\t")]
    if "--write" in sys.argv:
        ref_root = Path(os.environ["SWOTGAT_REFERENCE"])
        for r in rows:
            r["md5"] = md5(ref_root / r["reference_path"])
        with open(table, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["reference_path", "md5", "smoke_path"], delimiter="\t", lineterminator="\n")
            w.writeheader(); w.writerows(rows)
        print("wrote", table)
        return 0
    ref_root = Path(os.environ["SWOTGAT_REFERENCE"]) if os.environ.get("SWOTGAT_REFERENCE") else None
    bad = 0
    for r in rows:
        got = WORK / (r["smoke_path"] or r["reference_path"])
        if not got.exists():
            print(f"MISSING     {r['reference_path']}"); bad += 1; continue
        if md5(got) == r["md5"]:
            print(f"identical   {r['reference_path']}"); continue
        ref = ref_root / r["reference_path"] if ref_root else EXPECTED / Path(r["reference_path"]).name
        if ref.exists() and same_values(got, ref):
            print(f"equal values {r['reference_path']} (bytes differ)"); continue
        print(f"DIFFERENT   {r['reference_path']}"); bad += 1
    print(f"{len(rows) - bad} of {len(rows)} outputs reproduce the reference")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
