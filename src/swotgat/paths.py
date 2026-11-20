"""Every machine-specific path used by the SWOT-GAT pipeline, in one place.

All scripts in this repository import their root folders from here, e.g.::

    from swotgat.paths import RESERVOIRS
    graph = f"{RESERVOIRS}/CONUS/metadata/graphs_conus_v35.json"

The values come from ``config/paths.env`` (copy ``config/paths.env.example``
and edit it). The file is plain ``export SWOTGAT_<NAME>=<path>`` lines, so
shell scripts read the same file with ``source config/paths.env``.

Lookup order for each name:

1. the environment variable ``SWOTGAT_<NAME>`` if it is set;
2. the file named by ``$SWOTGAT_PATHS_FILE`` if it is set;
3. ``config/paths.env`` at the repository root.

Importing a name that is not configured raises an error that says which line
to add, rather than failing later with a missing file.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[2])
"""Absolute path of this repository's root folder."""

_LINE = re.compile(r"^\s*(?:export\s+)?SWOTGAT_([A-Z0-9_]+)=(.*)$")


def _read_env_file(path: Path) -> dict[str, str]:
    """Parse ``export SWOTGAT_NAME=value`` lines (comments and blanks ignored)."""
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text().splitlines():
        m = _LINE.match(raw.split("#", 1)[0].rstrip())
        if not m:
            continue
        value = m.group(2).strip().strip('"').strip("'")
        # allow ${SWOTGAT_OTHER} references to names defined earlier in the file
        value = re.sub(r"\$\{SWOTGAT_([A-Z0-9_]+)\}",
                       lambda r: values.get(r.group(1), r.group(0)), value)
        values[m.group(1)] = value.rstrip("/")
    return values


def _load() -> dict[str, str]:
    path = Path(os.environ.get("SWOTGAT_PATHS_FILE", Path(REPO) / "config" / "paths.env"))
    values = _read_env_file(path)
    for key, value in os.environ.items():
        if key.startswith("SWOTGAT_") and key != "SWOTGAT_PATHS_FILE":
            values[key[len("SWOTGAT_"):]] = value.rstrip("/")
    values["REPO"] = REPO
    return values


_VALUES = _load()
globals().update(_VALUES)


def __getattr__(name: str) -> str:
    if name.startswith("_"):
        raise AttributeError(name)
    raise ImportError(
        f"swotgat.paths: '{name}' is not configured. Add the line\n"
        f"    export SWOTGAT_{name}=/your/path\n"
        f"to config/paths.env (see config/paths.env.example)."
    )


def as_dict() -> dict[str, str]:
    """Return every configured root as ``{NAME: path}``."""
    return dict(_VALUES)
