#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'Usage: %s OUTPUT_DIR [SVG_DIR] [WIDTH]\n' "$0"
  printf '  OUTPUT_DIR  Directory where PNG files will be written.\n'
  printf '  SVG_DIR     Directory containing SVG files. Default: figures\n'
  printf '  WIDTH       Export width in pixels. Default: 1800\n'
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -lt 1 || $# -gt 3 ]]; then
  usage
  exit $([[ $# -lt 1 || $# -gt 3 ]] && printf 1 || printf 0)
fi

out_dir=$1
svg_dir=${2:-figures}
width=${3:-1800}

if ! command -v inkscape >/dev/null 2>&1; then
  printf 'Error: inkscape is required but was not found on PATH.\n' >&2
  exit 1
fi

if [[ ! -d "$svg_dir" ]]; then
  printf 'Error: SVG directory does not exist: %s\n' "$svg_dir" >&2
  exit 1
fi

mkdir -p "$out_dir"

count=0
while IFS= read -r -d '' svg; do
  base=$(basename "$svg" .svg)
  png="$out_dir/$base.png"
  inkscape "$svg" --export-filename="$png" --export-width="$width"
  printf 'Wrote: %s\n' "$png"
  count=$((count + 1))
done < <(find "$svg_dir" -maxdepth 1 -type f -name '*.svg' -print0 | sort -z)

printf 'Exported %d SVG files to %s\n' "$count" "$out_dir"
