#!/bin/bash
set -euo pipefail

: "${MODE:?MODE is required: mid or mdd}"
: "${OUTNAME:?OUTNAME is required}"
POBS_MODE="${POBS_MODE:-observed-box}"
LIB_VMAG_MAX="${LIB_VMAG_MAX:--6.0}"

if [[ -n "${GALAXY_NAMES:-}" ]]; then
  read -r -a GALAXY_ARGS <<< "${GALAXY_NAMES}"
  GALAXY_LABEL="${GALAXY_LABEL:-${GALAXY_NAMES// /_}}"
elif [[ -n "${GALAXY:-}" ]]; then
  GALAXY_ARGS=("${GALAXY}")
  GALAXY_LABEL="${GALAXY_LABEL:-${GALAXY}}"
else
  echo "ERROR: set GALAXY for one galaxy or GALAXY_NAMES for a combined fit" >&2
  exit 1
fi

PIPE_ROOT="/g/data/jh2/jt4478/Tang26B/pipeline"
BUNDLE_DIR="${PIPE_ROOT}/bundled_pipeline"
VENV="${PIPE_ROOT}/.venv"
LOG_DIR="/g/data/jh2/jt4478/Tang26B/output_io"

LEGUS_CCT_ROOT="/g/data/jh2/jt4478/make_LEGUS_CCT"
LEGUS_TAB_DIR="/g/data/jh2/jt4478/Tang26B/cluster_data"
CLUSTER_SLUG_LIB_DIR="/g/data/jh2/jt4478/cluster_slug"
CLUSTER_SLUG_LIB_NAME="/g/data/jh2/jt4478/cluster_slug/tang"
NN_DIR="/g/data/jh2/jt4478/Tang26B/nn_models"
OUT_CHAIN_DIR="/g/data/jh2/jt4478/Tang26B/output_chains"

if [[ "${MODE}" != "mid" && "${MODE}" != "mdd" ]]; then
  echo "ERROR: MODE must be mid or mdd, got ${MODE}" >&2
  exit 1
fi

if [[ "${POBS_MODE}" != "observed-box" && "${POBS_MODE}" != "nn" && "${POBS_MODE}" != "hybrid" ]]; then
  echo "ERROR: POBS_MODE must be observed-box, nn, or hybrid, got ${POBS_MODE}" >&2
  exit 1
fi

MODE_FLAG=()
if [[ "${MODE}" == "mdd" ]]; then
  MODE_FLAG=(--mdd)
fi

SAFE_GALAXY="${GALAXY_LABEL//[^A-Za-z0-9_]/_}"
SAFE_OUTNAME="${OUTNAME%.h5}"
LOG_FILE="${LOG_DIR}/${SAFE_OUTNAME}_${SAFE_GALAXY}_${MODE}.log"

cd "${BUNDLE_DIR}"

module purge
module load python3/3.11.7
# If 3.11.7 is unavailable on the node, use: module load python3/3.11.0
module load hdf5/1.10.5

if [[ ! -f "${VENV}/bin/activate" ]]; then
  echo "ERROR: venv missing at ${VENV}" >&2
  echo "On login once: module load python3/3.11.7 && cd ${PIPE_ROOT} && python3 -m venv .venv && source .venv/bin/activate" >&2
  echo "  pip install -U pip && pip install torch --index-url https://download.pytorch.org/whl/cpu" >&2
  echo "  pip install \"scikit-learn>=1.6.1\" && pip install numpy scipy astropy emcee numexpr h5py matplotlib joblib slugpy" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${VENV}/bin/activate"

mkdir -p "${OUT_CHAIN_DIR}" "${LOG_DIR}"

{
  echo "GALAXY_NAMES=${GALAXY_ARGS[*]}"
  echo "MODE=${MODE}"
  echo "OUTNAME=${OUTNAME}"
  echo "POBS_MODE=${POBS_MODE}"
  echo "LIB_VMAG_MAX=${LIB_VMAG_MAX}"
  python3 -c "import torch, sklearn; print('torch', torch.__version__, torch.__file__); print('sklearn', sklearn.__version__, sklearn.__file__)"

  python "${BUNDLE_DIR}/analyze_catalog_mid_mdd.py" \
    "${CLUSTER_SLUG_LIB_NAME}" \
    "${CLUSTER_SLUG_LIB_DIR}/lib_mass.pdf" \
    "${CLUSTER_SLUG_LIB_DIR}/lib_time.pdf" \
    "${CLUSTER_SLUG_LIB_DIR}/lib_av.pdf" \
    --galaxy-names "${GALAXY_ARGS[@]}" \
    --catalog-glob "hlsp_legus*{galaxy_name}*.tab" \
    --legus-cct-root "${LEGUS_CCT_ROOT}" \
    --legus-tab-dir "${LEGUS_TAB_DIR}" \
    --cluster-slug-lib-dir "${CLUSTER_SLUG_LIB_DIR}" \
    --nn-dir "${NN_DIR}" \
    --output-mcmc-chains-dir "${OUT_CHAIN_DIR}" \
    --cattype LEGUS \
    --nwalkers 100 \
    --niter 4000 \
    --bwphot 0.05 \
    --bwphys 0.05 \
    --hybrid-range-margin 0.1 \
    --pobs-mode "${POBS_MODE}" \
    --lib-vmag-max "${LIB_VMAG_MAX}" \
    "${MODE_FLAG[@]}" \
    --outname "${OUTNAME}" \
    --verbose
} 2>&1 | tee "${LOG_FILE}"
