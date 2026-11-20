"""Checks that need no data: configurations, vendored model source, syntax.

1. The two model configurations, rendered with the authors' original project
   root, must have the md5 sums of the files the paper runs were trained from.
2. Every file of ``vendor/swot_ml/src`` must match ``provenance/files.tsv``
   (which records the md5 of the source tree the paper runs imported).
3. Every Python file of the repository must compile.
"""
from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ORIGINAL_ROOT = "/work/pi_casey_umass_edu/Owen/Reservoirs"   # only used to reproduce the config md5 sums
CONFIG_MD5 = {
    "05_training/configs/era5_conus_v35c5_S1_storage.json.template": "e91fd492fd5861fd3430869fc1583599",
    "05_training/configs/era5_conus_v35c5_S2_storage_s1.json.template": "55881f6336c7003399bfcbfc5d417920",
}


def md5(data: bytes) -> str:
    """Return the md5 sum of ``data`` as a hexadecimal string."""
    return hashlib.md5(data).hexdigest()


def main() -> int:
    """Run the three checks and return 1 if any failed, else 0."""
    bad = 0
    for rel, want in CONFIG_MD5.items():
        text = (REPO / rel).read_text().replace("${SWOTGAT_WORK}", ORIGINAL_ROOT)
        got = md5(text.encode())
        print(("ok   " if got == want else "FAIL ") + f"config {Path(rel).name}: {got}")
        bad += got != want
    n = 0
    with open(REPO / "provenance" / "files.tsv", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if row["release_path"].startswith("vendor/swot_ml/src/"):
                n += 1
                if md5((REPO / row["release_path"]).read_bytes()) != row["original_md5"]:
                    print("FAIL vendored model file changed:", row["release_path"])
                    bad += 1
    print(f"ok   {n} vendored model files match the source the paper runs imported" if not bad else "")
    count = 0
    for path in sorted(REPO.rglob("*.py")):
        if ".git" in path.parts:
            continue
        try:
            compile(path.read_text(), str(path), "exec")
            count += 1
        except SyntaxError as err:
            print("FAIL does not compile:", path.relative_to(REPO), err)
            bad += 1
    print(f"ok   {count} Python files compile")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
