#!/usr/bin/env python3
"""Replace hard-coded absolute paths with names from ``swotgat.paths``.

This is the tool that produced the path edits recorded in PROVENANCE.md. It
is kept in the repository so the edits can be audited and repeated.

For Python files the rewrite is done on tokens, never on raw text, so that
the result is provably equivalent:

* a plain string literal that contains a known root, e.g.
  ``"/work/.../Reservoirs/CONUS/metadata/x.json"``, becomes the f-string
  ``f"{RESERVOIRS}/CONUS/metadata/x.json"``;
* the literal part of an existing f-string gets ``{RESERVOIRS}`` in place of
  the root;
* comments and docstrings get the text ``$SWOTGAT_RESERVOIRS``;
* one ``from swotgat.paths import ...`` line is added.

``--verify`` then parses the original and the rewritten file, substitutes the
original root values back for the imported names, and requires the two syntax
trees to be identical. A file that passes therefore computes exactly what the
original computed whenever ``config/paths.env`` holds the original roots.

Shell files get ``${SWOTGAT_<NAME>}``; lines where the shell would not expand
it (``#SBATCH`` directives, single-quoted text) are reported for hand review.

Usage:
    rewrite_paths.py --roots roots.tsv [--verify-only] ORIG NEW
        ORIG  the unmodified source file
        NEW   where to write the rewritten file (may equal a copy of ORIG)
"""
from __future__ import annotations

import argparse
import ast
import io
import re
import sys
import tokenize
from pathlib import Path

IMPORT_MODULE = "swotgat.paths"


def load_roots(tsv: str) -> list[tuple[str, str]]:
    """Read ``NAME<TAB>/absolute/prefix`` lines; longest prefix first."""
    roots = []
    for line in Path(tsv).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, prefix = line.split("\t")
        roots.append((name, prefix.rstrip("/")))
    roots.sort(key=lambda r: -len(r[1]))
    return roots


# --------------------------------------------------------------------------
# Python
# --------------------------------------------------------------------------
def _docstring_positions(tree: ast.AST) -> set[tuple[int, int]]:
    """(line, col) of every string used as a bare expression statement."""
    pos = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            pos.add((node.value.lineno, node.value.col_offset))
    return pos


def _sub_text(text: str, roots, fmt: str) -> tuple[str, set[str]]:
    used = set()
    for name, prefix in roots:
        pat = re.compile(re.escape(prefix) + r"(?![A-Za-z0-9_\-.])")
        if pat.search(text):
            used.add(name)
            text = pat.sub(fmt.format(name=name), text)
    return text, used


def rewrite_python(src: str, roots) -> tuple[str, set[str], list[str]]:
    """Return (new source, names used, warnings)."""
    warnings: list[str] = []
    if not any(prefix in src for _, prefix in roots):
        return src, set(), warnings
    tree = ast.parse(src)
    docpos = _docstring_positions(tree)
    lines = src.splitlines(keepends=True)
    starts = [0]
    for ln in lines:
        starts.append(starts[-1] + len(ln))

    def off(row: int, col: int) -> int:
        # tokenize columns are in characters for str input
        return starts[row - 1] + col

    edits: list[tuple[int, int, str]] = []
    used_code: set[str] = set()
    toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    for tok in toks:
        s, e = off(*tok.start), off(*tok.end)
        text = src[s:e]
        if not any(prefix in text for _, prefix in roots):
            continue
        if tok.type == tokenize.COMMENT:
            new, _ = _sub_text(text, roots, "$SWOTGAT_{name}")
            edits.append((s, e, new))
        elif tok.type == tokenize.STRING:
            m = re.match(r"(?i)([rbu]*)('''|\"\"\"|'|\")", text)
            pre, quote = m.group(1), m.group(2)
            if tok.start in docpos:
                new, _ = _sub_text(text, roots, "$SWOTGAT_{name}")
                edits.append((s, e, new))
                continue
            if "b" in pre.lower():
                warnings.append(f"line {tok.start[0]}: bytes literal with a root; not rewritten")
                continue
            body = text[len(pre) + len(quote):-len(quote)]
            body = body.replace("{", "{{").replace("}", "}}")
            body, used = _sub_text(body, roots, "{{{name}}}")
            used_code |= used
            newpre = pre.replace("u", "").replace("U", "")
            edits.append((s, e, f"f{newpre}{quote}{body}{quote}"))
        elif tok.type == getattr(tokenize, "FSTRING_MIDDLE", -1):
            new, used = _sub_text(text, roots, "{{{name}}}")
            used_code |= used
            edits.append((s, e, new))
        else:
            warnings.append(f"line {tok.start[0]}: root inside token type {tok.type}; not rewritten")

    out = src
    for s, e, new in sorted(edits, reverse=True):
        out = out[:s] + new + out[e:]

    if used_code:
        out = _add_import(out, sorted(used_code))
    return out, used_code, warnings


def _add_import(src: str, names: list[str]) -> str:
    """Insert the import after the module docstring and ``__future__`` imports."""
    tree = ast.parse(src)
    line = 0  # insert before this 0-based line index
    body = tree.body
    i = 0
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        line = body[0].end_lineno
        i = 1
    while i < len(body) and isinstance(body[i], ast.ImportFrom) and body[i].module == "__future__":
        line = body[i].end_lineno
        i += 1
    lines = src.splitlines(keepends=True)
    if line == 0:
        # keep a shebang / encoding line first
        while line < len(lines) and lines[line].startswith("#"):
            line += 1
    stmt = f"from {IMPORT_MODULE} import {', '.join(names)}\n"
    lines.insert(line, stmt)
    return "".join(lines)


class _Fold(ast.NodeTransformer):
    """Fold f-string pieces that are root names back to constants."""

    def __init__(self, values: dict[str, str]):
        self.values = values

    def visit_JoinedStr(self, node: ast.JoinedStr):
        self.generic_visit(node)
        parts: list[ast.expr] = []
        for v in node.values:
            if isinstance(v, ast.FormattedValue) and isinstance(v.value, ast.Name) \
                    and v.value.id in self.values and v.conversion == -1 and v.format_spec is None:
                v = ast.Constant(self.values[v.value.id])
            if isinstance(v, ast.Constant) and parts and isinstance(parts[-1], ast.Constant):
                parts[-1] = ast.Constant(parts[-1].value + v.value)
            else:
                parts.append(v)
        if not parts:
            return ast.Constant("")
        if len(parts) == 1 and isinstance(parts[0], ast.Constant):
            return ast.Constant(parts[0].value)
        node.values = parts
        return node


def _normalise(src: str, values: dict[str, str]) -> str:
    tree = ast.parse(src)
    keep = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == IMPORT_MODULE:
            continue
        keep.append(node)
    tree.body = keep
    tree = _Fold(values).visit(tree)
    # string-only statements (docstrings) carry no behaviour: drop them, so that
    # adding, moving or rewording a docstring does not count as a code change
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            block = getattr(node, field, None)
            if isinstance(block, list):
                setattr(node, field, [s for s in block if not (
                    isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant) and isinstance(s.value.value, str))])
    return ast.dump(tree, include_attributes=False)


def verify_python(orig: str, new: str, roots) -> bool:
    """Return True if ``new`` has the syntax tree of ``orig`` once the root names are put back."""
    values = {name: prefix for name, prefix in roots}
    return _normalise(orig, {}) == _normalise(new, values)


# --------------------------------------------------------------------------
# Shell
# --------------------------------------------------------------------------
def rewrite_shell(src: str, roots) -> tuple[str, set[str], list[str]]:
    """Replace roots in a shell script by ``${SWOTGAT_<NAME>}``; return (text, names used, warnings)."""
    warnings, used = [], set()
    out_lines = []
    heredoc_quoted = None
    for n, line in enumerate(src.splitlines(keepends=True), 1):
        has = any(prefix in line for _, prefix in roots)
        if heredoc_quoted is not None:
            if line.strip() == heredoc_quoted:
                heredoc_quoted = None
            elif has:
                warnings.append(f"line {n}: root inside a quoted here-document; not expanded by the shell")
        else:
            m = re.search(r"<<-?\s*(['\"])(\w+)\1", line)
            if m:
                heredoc_quoted = m.group(2)
        if has:
            if line.lstrip().startswith("#SBATCH"):
                warnings.append(f"line {n}: #SBATCH directive holds a root; Slurm does not expand variables")
            else:
                idx = min(line.find(p) for _, p in roots if p in line)
                if line[:idx].count("'") % 2 == 1 and not line.lstrip().startswith("#"):
                    warnings.append(f"line {n}: root inside single quotes; check expansion")
            line, u = _sub_text(line, roots, "${{SWOTGAT_{name}}}")
            used |= u
        out_lines.append(line)
    return "".join(out_lines), used, warnings


# --------------------------------------------------------------------------
def main() -> int:
    """Rewrite or verify one file as given on the command line."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", required=True)
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("orig")
    ap.add_argument("new")
    a = ap.parse_args()
    roots = load_roots(a.roots)
    orig = Path(a.orig).read_text()
    is_py = a.orig.endswith(".py")
    if a.verify_only:
        new = Path(a.new).read_text()
    else:
        new, used, warns = (rewrite_python if is_py else rewrite_shell)(orig, roots)
        Path(a.new).write_text(new)
        for w in warns:
            print(f"WARN {a.new}: {w}")
    if is_py:
        ok = verify_python(orig, new, roots)
        print(("VERIFIED " if ok else "MISMATCH ") + a.new)
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
