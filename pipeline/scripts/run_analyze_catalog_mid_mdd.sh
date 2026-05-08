#!/usr/bin/env bash
# Run the bundled MCMC driver with PYTHONPATH set (Gadi / tarball friendly).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/bundled_pipeline${PYTHONPATH:+:${PYTHONPATH}}"
exec python "${ROOT}/bundled_pipeline/analyze_catalog_mid_mdd.py" "$@"
