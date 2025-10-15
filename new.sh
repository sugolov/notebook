#!/usr/bin/env bash
set -euo pipefail
d="$(cd -- "$(dirname -- "$0")" && pwd)"
dest="$1"

[ -e "$dest" ] && { echo "directory exists: $dest"; exit 1; }

mkdir -p "$dest"
cp "$d/template/main.tex" "$d/template/bib.bib" "$dest/"

