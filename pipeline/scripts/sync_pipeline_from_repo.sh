#!/usr/bin/env bash
# Copy pipeline Python sources from the parent slugfiles repo into bundled_pipeline/.
# Run from anywhere; resolves repo root as parent of new_pipeline_slug_legus.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${PKG_ROOT}/.." && pwd)"
SRC="${REPO_ROOT}/python3_sc"
if [[ ! -d "$SRC" ]]; then
  echo "error: expected python3_sc at ${SRC}" >&2
  exit 1
fi
for f in \
  analyze_catalog_mid_mdd.py \
  catalog_readers.py \
  completeness_calculator.py \
  clean_legus.py \
  completeness_io.py
do
  cp -v "${SRC}/${f}" "${PKG_ROOT}/bundled_pipeline/${f}"
done
echo "Synced bundled_pipeline from ${SRC}"

FIX_SRC="${REPO_ROOT}/tests/data/legus_hlsp"
if [[ -d "${FIX_SRC}" ]]; then
  mkdir -p "${PKG_ROOT}/fixtures/legus_hlsp"
  rsync -a --delete "${FIX_SRC}/" "${PKG_ROOT}/fixtures/legus_hlsp/"
  echo "Synced fixtures/legus_hlsp from ${FIX_SRC}"
else
  echo "note: no ${FIX_SRC}; fixtures not updated"
fi
