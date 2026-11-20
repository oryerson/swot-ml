#!/usr/bin/env python3
"""Check Python files against the docstring conventions of PEP 257.

Usage::

    python tools/check_docstrings.py [--summary] PATH [PATH ...]

Each PATH is a file or a folder (searched for ``*.py``). One line is printed
per violation, as ``file:line: CODE message``; the exit status is 1 if any
were found. The codes follow the numbering of the ``pydocstyle`` tool:

==== =====================================================================
D100 module has no docstring
D101 public class has no docstring
D102 public method has no docstring
D103 public function has no docstring
D105 special method (``__repr__`` ...) has no docstring
D107 ``__init__`` has no docstring
D200 a docstring that fits on one line is spread over several
D201 blank line before a function docstring
D202 blank line after a function docstring
D204 no blank line after a class docstring
D205 no blank line between the summary line and the description
D207 a docstring line is indented less than its quotes
D209 closing quotes of a multi-line docstring are not on their own line
D210 whitespace around the text of a one-line docstring, or before the summary
D211 blank line before a class docstring
D300 docstring does not use triple double quotes
D301 docstring has a backslash but is not a raw string
D400 summary line does not end with a period
D401 summary line is not in the imperative mood ("Return", not "Returns")
D402 summary line repeats the function's signature
D403 summary line does not start with a capital letter
D419 docstring is empty
==== =====================================================================

"Public" means a name without a leading underscore, defined at module level
or in a public class. Functions defined inside other functions are not
checked, as in PEP 257.
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

# Third-person verbs that often start a summary line by mistake (D401).
NOT_IMPERATIVE = {
    "returns", "builds", "reads", "writes", "computes", "creates", "makes", "gets", "sets", "loads",
    "saves", "checks", "runs", "adds", "removes", "applies", "converts", "finds", "fits", "draws",
    "plots", "scores", "parses", "yields", "generates", "calculates", "extracts", "downloads",
    "fetches", "merges", "joins", "filters", "selects", "splits", "maps", "matches", "tests",
    "prints", "updates", "copies", "lists", "counts", "sums", "calls", "does", "gives", "takes",
    "uses", "puts", "keeps", "drops", "picks", "looks", "tries", "wraps", "opens", "sorts",
    "this", "function", "method", "helper",
}


def _docstring_node(node: ast.AST) -> ast.Expr | None:
    """Return the statement that holds ``node``'s docstring, if it has one."""
    body = getattr(node, "body", None)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
            and isinstance(body[0].value.value, str):
        return body[0]
    return None


def _check_text(kind: str, name: str, expr: ast.Expr, lines: list[str], first_body_line: int | None,
                def_line: int) -> list[tuple[int, str, str]]:
    """Check the form of one docstring and return ``(line, code, message)`` tuples."""
    out = []
    lo, hi = expr.lineno, expr.end_lineno
    raw = "\n".join(lines[lo - 1:hi])
    start = lines[lo - 1][expr.col_offset:]
    m = re.match(r"([rRuUbB]*)(\"\"\"|'''|\"|')", start)
    prefix, quote = m.group(1), m.group(2)
    indent = expr.col_offset
    text = expr.value.value
    if quote != '"""':
        out.append((lo, "D300", 'use """triple double quotes"""'))
    if "\\" in raw and "r" not in prefix.lower():
        out.append((lo, "D301", 'use r""" if the docstring contains a backslash'))
    if not text.strip():
        out.append((lo, "D419", "docstring is empty"))
        return out
    doc_lines = text.split("\n")
    multi = hi > lo
    summary = doc_lines[0].strip() if doc_lines[0].strip() else (doc_lines[1].strip() if len(doc_lines) > 1 else "")
    summary_index = 0 if doc_lines[0].strip() else 1
    if doc_lines[0] != doc_lines[0].lstrip():
        out.append((lo, "D210", "no whitespace before the docstring text"))
    if not multi and text != text.strip():
        out.append((lo, "D210", "no whitespace around the text of a one-line docstring"))
    non_empty = [l for l in doc_lines if l.strip()]
    if multi and len(non_empty) == 1 and len(lines[lo - 1][:indent]) + len(prefix) + 6 + len(summary) <= 100:
        out.append((lo, "D200", "a one-line docstring should fit on one line with its quotes"))
    if multi and len(non_empty) > 1:
        after = doc_lines[summary_index + 1] if len(doc_lines) > summary_index + 1 else ""
        if after.strip():
            out.append((lo + summary_index, "D205", "one blank line needed between the summary line and the description"))
        if doc_lines[-1].strip():
            out.append((hi, "D209", "closing quotes of a multi-line docstring go on a line of their own"))
        for k, l in enumerate(lines[lo:hi], lo + 1):
            if l.strip() and len(l) - len(l.lstrip()) < indent:
                out.append((k, "D207", "docstring line is indented less than its quotes"))
                break
    if summary and not summary.endswith((".", "?", "!", "::", ":")) or summary.endswith((":",)) and not summary.endswith("::"):
        out.append((lo + summary_index, "D400", "summary line should end with a period"))
    first_word = re.split(r"[\s(,:]", summary, 1)[0] if summary else ""
    if first_word and first_word[0].isalpha() and first_word[0].islower() and first_word.isalpha():
        out.append((lo + summary_index, "D403", "summary line should start with a capital letter"))
    if kind in ("function", "method") and first_word.lower().strip("`") in NOT_IMPERATIVE:
        out.append((lo + summary_index, "D401", f'summary line should be imperative ("{first_word}" is not)'))
    if kind in ("function", "method") and re.match(re.escape(name) + r"\s*\(", summary):
        out.append((lo + summary_index, "D402", "summary line should not repeat the signature"))
    if kind in ("function", "method"):
        if lo - def_line > 1 and all(not lines[k].strip() for k in range(def_line, lo - 1)) \
                and lines[def_line - 1].rstrip().endswith(":"):
            out.append((lo, "D201", "no blank line before a function docstring"))
        if first_body_line and first_body_line - hi > 1 and not lines[hi].strip():
            nxt = lines[first_body_line - 1].lstrip()
            if not nxt.startswith(("def ", "class ", "@", "async def ")):
                out.append((hi, "D202", "no blank line after a function docstring"))
    if kind == "class":
        if lo - def_line > 1 and not lines[lo - 2].strip():
            out.append((lo, "D211", "no blank line before a class docstring"))
        if first_body_line and first_body_line - hi == 1:
            out.append((hi, "D204", "one blank line needed after a class docstring"))
    return out


def check_source(src: str) -> list[tuple[int, str, str]]:
    """Return every violation in one source text as ``(line, code, message)``."""
    tree = ast.parse(src)
    lines = src.splitlines()
    out: list[tuple[int, str, str]] = []

    def visit(node: ast.AST, kind: str, name: str, public: bool, missing: str) -> None:
        expr = _docstring_node(node)
        def_line = getattr(node, "lineno", 1)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # the line that ends the header (a signature can span several lines)
            def_line = (node.body[0].lineno - 1) if node.body[0].lineno > node.lineno else node.lineno
            while def_line > node.lineno and not lines[def_line - 1].strip():
                def_line -= 1
        if expr is None:
            if public:
                out.append((getattr(node, "lineno", 1), missing, f"{kind} {name} has no docstring".replace("module  ", "module ")))
        else:
            nxt = node.body[1].lineno if len(node.body) > 1 else None
            if kind != "module":
                out.extend(_check_text(kind, name, expr, lines, nxt, def_line))
            else:
                out.extend(_check_text(kind, name, expr, lines, None, 0))
        if isinstance(node, (ast.Module, ast.ClassDef)):
            for child in node.body:
                if isinstance(child, ast.ClassDef):
                    visit(child, "class", child.name, public and not child.name.startswith("_"), "D101")
                elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    n = child.name
                    if isinstance(node, ast.ClassDef):
                        code = "D107" if n == "__init__" else "D105" if n.startswith("__") and n.endswith("__") else "D102"
                        pub = public and (not n.startswith("_") or (n.startswith("__") and n.endswith("__")))
                        visit(child, "method", n, pub, code)
                    else:
                        visit(child, "function", n, not n.startswith("_"), "D103")

    visit(tree, "module", "", True, "D100")
    return sorted(out)


def main() -> int:
    """Run the checker on the paths given on the command line."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--summary", action="store_true", help="print only the number of violations per code")
    a = ap.parse_args()
    files = []
    for p in map(Path, a.paths):
        files += sorted(p.rglob("*.py")) if p.is_dir() else [p]
    total: dict[str, int] = {}
    bad_files = 0
    for f in files:
        try:
            found = check_source(f.read_text())
        except SyntaxError as err:
            print(f"{f}: cannot parse: {err}")
            return 2
        bad_files += bool(found)
        for line, code, msg in found:
            total[code] = total.get(code, 0) + 1
            if not a.summary:
                print(f"{f}:{line}: {code} {msg}")
    if a.summary or total:
        print(f"{len(files)} files, {bad_files} with violations: " + (", ".join(f"{c} {n}" for c, n in sorted(total.items())) or "none"))
    else:
        print(f"{len(files)} files comply with PEP 257")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
