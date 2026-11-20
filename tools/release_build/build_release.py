#!/usr/bin/env python3
"""Copy the original project files into the release tree and record provenance.

This script documents (and can repeat) how the release was assembled from the
working folders on the Unity cluster. It is not part of the science pipeline.

Inputs, all in this folder:

``manifest.tsv``
    ``release_path <TAB> original_path <TAB> mode <TAB> note <TAB> exclude <TAB> deploy_as``.
    A ``release_path`` ending in ``/`` is a folder row: ``original_path`` is
    then a glob and ``exclude`` an optional regular expression of files to skip.
    ``mode`` is ``auto`` (rewrite hard-coded roots, see ``rewrite_paths.py``)
    or ``raw`` (byte-identical copy). ``$R`` in ``original_path`` stands for
    the original project root (first line of ``roots.tsv``).
``roots.tsv``
    ``NAME <TAB> /absolute/prefix``: the roots replaced by ``swotgat.paths``.
``relocations.tsv``
    ``original_path_or_dir <TAB> replacement``: references to code that moved.
    ``replacement`` is ``REPO/<path in repo>`` or ``ENV/<SWOTGAT variable>``.

Commands::

    build_release.py copy [--force] [--only PREFIX]   # originals -> release tree
    build_release.py edits                            # apply hand_edits.py
    build_release.py provenance                       # write provenance/files.tsv

``copy`` never overwrites an existing release file unless ``--force`` is given,
so hand edits made after the first copy are kept. ``provenance`` re-checks
every file against its original and classifies it:

``identical``     byte-for-byte copy.
``paths-only``    the Python syntax tree equals the original once the path
                  names are replaced by the original roots (proved per file).
``paths-only*``   shell or text file; differs only in lines that held a path.
``edited``        changed beyond paths; the manifest note says what and why.
``new``           written for the release.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import rewrite_paths as rw  # noqa: E402

PY_ENV_PREFIX = "SWOTGAT_"


def md5(path: Path) -> str:
    """Return the md5 sum of a file as a hexadecimal string."""
    return hashlib.md5(path.read_bytes()).hexdigest()


def read_tsv(name: str, ncol: int) -> list[list[str]]:
    """Read a tab-separated file of this folder as rows of ``ncol`` stripped cells, skipping comments."""
    rows = []
    path = HERE / name
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        parts += [""] * (ncol - len(parts))
        rows.append([p.strip() for p in parts[:ncol]])
    return rows


def load():
    """Return the path roots, the expanded manifest and the relocations."""
    roots = rw.load_roots(str(HERE / "roots.tsv"))
    r_value = dict(roots)["WORK"]
    manifest = [(d, s, m, n) for d, s, m, n, _ in expand_manifest(r_value)]
    reloc = [(a.replace("$R", r_value), b) for a, b in read_tsv("relocations.tsv", 2)]
    reloc.sort(key=lambda r: -len(r[0]))
    return roots, manifest, reloc


def expand_manifest(r_value: str) -> list[tuple[str, str, str, str, str]]:
    """Return (release_path, original_path, mode, note, deploy_as) with folder rows expanded."""
    out = []
    for dest, src, mode, note, exclude, deploy_as in read_tsv("manifest.tsv", 6):
        src = src.replace("$R", r_value)
        if dest.endswith("/"):          # folder row: src is a glob, dest a folder
            base = Path(src.split("*", 1)[0])
            if not src.split("*", 1)[0].endswith("/"):
                base = base.parent
            pattern = str(Path(src).relative_to(base))
            for match in sorted(base.glob(pattern)):
                rel = str(match.relative_to(base))
                if not match.is_file() or "__pycache__" in rel:
                    continue
                if exclude and re.search(exclude, rel):
                    continue
                out.append((dest + rel, str(match), mode or "auto", note, ""))
        else:
            out.append((dest, src, mode or "auto", note, deploy_as))
    return out


def _root_form(path: str, roots, shell: bool) -> str | None:
    """Write an original absolute path the way the root rewrite leaves it."""
    for name, prefix in roots:
        if path == prefix or path.startswith(prefix + "/"):
            rest = path[len(prefix):]
            return (f"${{{PY_ENV_PREFIX}{name}}}" if shell else f"{{{name}}}") + rest
    return None


def _target_form(target: str, shell: bool) -> str:
    kind, _, rest = target.partition("/")
    if kind == "REPO":
        return (f"${{{PY_ENV_PREFIX}REPO}}" if shell else "{REPO}") + (f"/{rest}" if rest else "")
    if kind == "ENV":
        return f"${{{PY_ENV_PREFIX}{rest}}}" if shell else f"{{{rest}}}"
    raise ValueError(target)


_BOUNDARY = r"(?![A-Za-z0-9_\-.])"


def relocate(text: str, roots, reloc, shell: bool, inverse: bool = False) -> str:
    """Replace references to code that moved; with ``inverse`` undo the replacement."""
    pairs = []
    for orig, target in reloc:
        a = _root_form(orig, roots, shell)
        if a is None:
            continue
        pairs.append((a, _target_form(target, shell)))
    if inverse:
        pairs = sorted(((b, a) for a, b in pairs), key=lambda p: -len(p[0]))
    for a, b in pairs:
        text = re.sub(re.escape(a) + _BOUNDARY, lambda m: b, text)
    return text


_IMPORT = re.compile(r"^from swotgat\.paths import .*$", re.M)


def fix_import(text: str, names: list[str]) -> str:
    """Make the swotgat.paths import list match the names the file uses."""
    used = sorted(n for n in names if re.search(r"\{" + n + r"\}", text))
    line = f"from swotgat.paths import {', '.join(used)}"
    if _IMPORT.search(text):
        return _IMPORT.sub(line, text, count=1) if used else text
    return text


def transform(text: str, dest: Path, mode: str, roots, reloc) -> tuple[str, list[str]]:
    """The automatic edit: root names, then relocations. Returns (text, warnings)."""
    if mode == "template":              # text file: roots -> ${SWOTGAT_<NAME>} placeholders
        for name, prefix in roots:
            text = text.replace(prefix, f"${{{PY_ENV_PREFIX}{name}}}")
        return text, []
    shell = dest.suffix == ".sh"
    new, _, warns = (rw.rewrite_shell if shell else rw.rewrite_python)(text, roots)
    if not shell and not rw.verify_python(text, new, roots):
        raise SystemExit(f"root rewrite of {dest} is not equivalent; fix the tool")
    new = relocate(new, roots, reloc, shell)
    if not shell:
        new = fix_import(new, [n for n, _ in roots] + ["REPO", "PYTHON", "DELINEATOR_PYTHON"])
    return new, warns


def build_one(dest: Path, src: Path, mode: str, roots, reloc) -> list[str]:
    """Write one release file from its original and return the rewrite warnings."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if mode == "raw" or (mode != "template" and dest.suffix not in (".py", ".sh")):
        dest.write_bytes(src.read_bytes())
        return []
    new, warns = transform(src.read_text(), dest, mode, roots, reloc)
    dest.write_text(new)
    dest.chmod(0o755 if dest.suffix == ".sh" else 0o644)
    return warns


def classify(dest: Path, src: Path, mode: str, roots, reloc) -> tuple[str, int]:
    """Return (class, number of changed lines).

    ``paths-only`` means: the release file has the same Python syntax tree as
    the automatic edit of the original (docstrings and comments aside). The
    automatic edit itself is proved equivalent to the original in transform().
    """
    if dest.read_bytes() == src.read_bytes():
        return "identical", 0
    try:
        a = src.read_text()
        b = dest.read_text()
    except UnicodeDecodeError:
        return "edited", -1
    changed = sum(1 for l in difflib.ndiff(a.splitlines(), b.splitlines()) if l[:1] in "+-")
    if mode == "raw":
        return "edited", changed
    expected, _ = transform(a, dest, mode, roots, reloc)
    if dest.suffix == ".py":
        try:
            same = rw._normalise(expected, {}) == rw._normalise(b, {})
        except SyntaxError:
            same = False
        return ("paths-only" if same else "edited"), changed
    return ("paths-only*" if expected == b else "edited"), changed


def cmd_copy(args) -> None:
    """Copy every manifest file that is not in the release tree yet (all files with ``--force``)."""
    roots, manifest, reloc = load()
    n = 0
    for dest, src, mode, _ in manifest:
        if src in ("", "-"):
            continue
        if args.only and not dest.startswith(args.only):
            continue
        d, s = REPO / dest, Path(src)
        if not s.exists():
            print(f"MISSING {s}")
            continue
        if d.exists() and not args.force:
            continue
        for w in build_one(d, s, mode, roots, reloc):
            print(f"WARN {dest}: {w}")
        n += 1
    print(f"copied {n} files")


def deploy_target(dest: str, src: str, deploy_as: str, roots) -> str:
    """Where tools/deploy.py puts the file: ``ROOT:relative/path`` or '' (not deployed)."""
    if deploy_as:
        return deploy_as if ":" in deploy_as else f"WORK:{deploy_as}"
    if dest.startswith(("vendor/", "environment/", "tools/", "src/", "bin/", "config/", "docs/", "tests/")):
        return ""
    for name, prefix in roots:
        if src.startswith(prefix + "/"):
            return f"{name}:{src[len(prefix) + 1:]}"
    return ""


def cmd_edits(args) -> None:
    """Apply hand_edits.EDITS; an edit already applied is skipped."""
    from hand_edits import EDITS
    for dest, old, new, _ in EDITS:
        path = REPO / dest
        text = path.read_text()
        if old.startswith("re:"):
            pattern = re.compile(old[3:])
            hits = len(pattern.findall(text))
            if hits == 0:
                continue                      # already applied
            if hits != 1:
                raise SystemExit(f"hand edit pattern matches {hits} times in {dest}")
            path.write_text(pattern.sub(new, text))
        else:
            applied = (new in text) if old in new else (old not in text)
            if applied:
                continue
            if text.count(old) != 1:
                raise SystemExit(f"hand edit does not apply cleanly to {dest}: {old[:60]!r} occurs {text.count(old)} times")
            path.write_text(text.replace(old, new))
        print(f"edited {dest}")


def cmd_provenance(args) -> None:
    """Classify every release file against its original and write ``provenance/files.tsv``."""
    roots, manifest, reloc = load()
    r_value = dict(roots)["WORK"]
    deploy_as = {d: x for d, _, _, _, x in expand_manifest(r_value)}
    from hand_edits import EDITS
    reasons: dict[str, list[str]] = {}
    for dest, _, _, why in EDITS:
        if why not in reasons.setdefault(dest, []):
            reasons[dest].append(why)
    rows = []
    counts: dict[str, int] = {}
    listed = set()
    for dest, src, mode, note in manifest:
        d = REPO / dest
        listed.add(dest)
        if not d.exists():
            print(f"NOT IN RELEASE: {dest}")
            continue
        if src in ("", "-"):
            cls, changed, osum, oshow = "new", 0, "", "(written for the release)"
            src = ""
        else:
            s = Path(src)
            cls, changed = classify(d, s, mode, roots, reloc)
            osum, oshow = md5(s), src.replace(r_value, "$R")
        counts[cls] = counts.get(cls, 0) + 1
        if cls == "edited" and dest in reasons:
            note = "; ".join(([note] if note else []) + reasons[dest])
        if cls == "edited" and not note:
            print(f"EDITED WITHOUT NOTE: {dest}")
        rows.append((dest, oshow, osum, md5(d), cls, changed, deploy_target(dest, src, deploy_as.get(dest, ""), roots), note))
    out = ["release_path\toriginal_path\toriginal_md5\trelease_md5\tclass\tlines_changed\tdeploy_to\tnote"]
    out += ["\t".join(str(x) for x in r) for r in rows]
    (REPO / "provenance" ).mkdir(exist_ok=True)
    (REPO / "provenance" / "files.tsv").write_text("\n".join(out) + "\n")
    print("classes:", counts, "total", len(rows))
    # every tracked file must be in the manifest
    skip = (".git/", "provenance/files.tsv")
    for p in sorted(REPO.rglob("*")):
        rel = str(p.relative_to(REPO))
        if p.is_file() and rel not in listed and not rel.startswith(skip) and "__pycache__" not in rel:
            print(f"NOT IN MANIFEST: {rel}")


def main() -> None:
    """Run the sub-command given on the command line."""
    if sys.version_info < (3, 12):  # rewrite_paths relies on the f-string tokens of Python 3.12
        raise SystemExit("build_release.py needs Python 3.12 or later")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("copy"); c.add_argument("--force", action="store_true"); c.add_argument("--only")
    c.set_defaults(fn=cmd_copy)
    e = sub.add_parser("edits"); e.set_defaults(fn=cmd_edits)
    p = sub.add_parser("provenance"); p.set_defaults(fn=cmd_provenance)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
