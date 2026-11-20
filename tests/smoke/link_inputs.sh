#!/bin/bash
# Link the read-only inputs listed in a file from an existing project tree
# into the (new, separate) work tree of this test.
#   usage: link_inputs.sh <existing project root> <list file>
# Only symbolic links are created, and only inside $SWOTGAT_WORK. Nothing is
# written to the existing tree. An entry that already exists is left alone.
set -euo pipefail
SRC=$1; LIST=$2
: "${SWOTGAT_WORK:?source config/paths.env first}"
[ "$(readlink -f "$SRC")" != "$(readlink -f "$SWOTGAT_WORK")" ] || { echo "the work tree must not be the existing project tree"; exit 1; }
grep -v '^#' "$LIST" | while read -r rel; do
  [ -n "$rel" ] || continue
  [ -e "$SRC/$rel" ] || { echo "MISSING in source: $rel"; continue; }
  [ -e "$SWOTGAT_WORK/$rel" ] && continue
  mkdir -p "$(dirname "$SWOTGAT_WORK/$rel")"
  ln -s "$SRC/$rel" "$SWOTGAT_WORK/$rel"
done
echo "inputs linked into $SWOTGAT_WORK"
