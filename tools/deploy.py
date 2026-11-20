#!/usr/bin/env python3
"""Lay the pipeline scripts out in the work tree, where they expect to run.

Why this step exists
--------------------
The project grew in one working folder that holds code and data side by side,
and many scripts find their inputs, their sibling modules and their output
folder relative to their own location. Rewriting all of that would risk
changing results. Instead, the repository keeps the code organised by pipeline
stage (``01_graph`` ... ``08_figures``) and this script copies each file to the
place in the work tree (``$SWOTGAT_WORK``) from which it originally ran.
``provenance/files.tsv`` holds the mapping (column ``deploy_to``).

After deploying, run any step with ``bin/run <path in this repository>``.

Usage::

    python tools/deploy.py            # copy what is missing, report what differs
    python tools/deploy.py --check    # report only, write nothing
    python tools/deploy.py --force    # also replace files that differ

Safety: an existing file that differs from the release copy is never replaced
unless ``--force`` is given. Small frozen input tables (``*/inputs/*``) are
never replaced at all, because a later pipeline step may have rebuilt them.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from swotgat import paths  # noqa: E402


def targets() -> list[tuple[Path, Path, bool]]:
    """Yield (release file, deployed path, is_template) for every deployed file."""
    roots = paths.as_dict()
    out = []
    with open(REPO / "provenance" / "files.tsv", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if not row["deploy_to"]:
                continue
            root, rel = row["deploy_to"].split(":", 1)
            if root not in roots:
                raise SystemExit(f"config/paths.env does not define SWOTGAT_{root} (needed for {row['release_path']})")
            is_template = row["release_path"].endswith(".template")
            out.append((REPO / row["release_path"], Path(roots[root]) / rel, is_template))
    return out


def render(text: str) -> str:
    """Replace ``${SWOTGAT_NAME}`` placeholders with the configured paths."""
    roots = paths.as_dict()
    return re.sub(r"\$\{SWOTGAT_([A-Z0-9_]+)\}", lambda m: roots[m.group(1)], text)


def generated() -> list[tuple[Path, str]]:
    """Small pointer files the scripts read, written from the configuration."""
    work = Path(paths.WORK)
    src = f"{paths.REPO}/vendor/swot_ml/src\n"
    return [(work / "CONUS_storage_fork/eval/v35_launch/SRC", src)]


def inside_work_tree(dst: Path) -> bool:
    """False if a linked folder would send the write outside the configured roots."""
    parent = dst.parent
    while not parent.exists():
        parent = parent.parent
    real = parent.resolve()
    return any(real == Path(r).resolve() or Path(r).resolve() in real.parents for r in paths.as_dict().values()
               if r.startswith("/"))


def main() -> int:
    """Copy the deployable files into the work tree and report what was done."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="report only")
    ap.add_argument("--force", action="store_true", help="replace deployed files that differ")
    a = ap.parse_args()
    n_new = n_same = n_diff = 0
    jobs = [(dst, render(src.read_text()).encode() if tpl else src.read_bytes(), "/inputs/" in str(src))
            for src, dst, tpl in targets()]
    jobs += [(dst, text.encode(), False) for dst, text in generated()]
    # HALF v1.0 is imported from two folders of the work tree
    for folder in ("CONUS_storage_fork/eval/swotext/half_v1", "CONUS_storage_fork/eval/half_vd_review/release_copy"):
        for f in sorted((REPO / "vendor" / "half_v1").iterdir()):
            jobs.append((Path(paths.WORK) / folder / f.name, f.read_bytes(), False))
    for dst, data, frozen in jobs:
        if dst.exists():
            if dst.read_bytes() == data:
                n_same += 1
                continue
            n_diff += 1
            if frozen or not a.force or a.check:
                print(f"differs, kept: {dst}")
                continue
        else:
            n_new += 1
        if a.check:
            print(f"missing: {dst}")
            continue
        if not inside_work_tree(dst):
            print(f"refused (folder is a link to somewhere else): {dst}")
            continue
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(data)
        except OSError as err:
            print(f"cannot write {dst}: {err}")
    print(f"{n_new} {'missing' if a.check else 'written'}, {n_same} already in place, {n_diff} differ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
