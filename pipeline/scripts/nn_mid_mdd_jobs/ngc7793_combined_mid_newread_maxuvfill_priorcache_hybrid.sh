#!/bin/bash
#PBS -P mk27
#PBS -q rsaa
#PBS -l walltime=24:00:00
#PBS -l ncpus=104
#PBS -l mem=100GB
#PBS -l jobfs=10GB
#PBS -l wd
#PBS -N nn_ngc7793_combined_newread
#PBS -j oe
#PBS -m bea
#PBS -M janet.tang@anu.edu.au
#PBS -o /scratch/jh2/jt4478/output/nn_ngc7793_combined_mid_newread_maxuvfill_priorcache_hybrid.o
#PBS -e /scratch/jh2/jt4478/output/nn_ngc7793_combined_mid_newread_maxuvfill_priorcache_hybrid.e
#PBS -l storage=scratch/jh2+gdata/jh2

set -euo pipefail

PIPE_ROOT="/g/data/jh2/jt4478/Tang26B/pipeline"
BUNDLE_DIR="${PIPE_ROOT}/bundled_pipeline"
VENV="${PIPE_ROOT}/.venv"
ROOT="/g/data/jh2/jt4478/Tang26B"
LOG_DIR="${ROOT}/output_io"
OUT_CHAIN_DIR="${ROOT}/output_chains"

LEGUS_CCT_ROOT="${LEGUS_CCT_ROOT:-/g/data/jh2/jt4478/make_LEGUS_CCT}"
LEGUS_TAB_DIR="${LEGUS_TAB_DIR:-${ROOT}/cluster_data}"
CLUSTER_SLUG_LIB_DIR="${CLUSTER_SLUG_LIB_DIR:-/g/data/jh2/jt4478/cluster_slug}"
CLUSTER_SLUG_LIB_NAME="${CLUSTER_SLUG_LIB_NAME:-/g/data/jh2/jt4478/cluster_slug/tang_padova}"
NN_DIR="${NN_DIR:-${ROOT}/nn_models}"
NITER="${NITER:-5000}"
NPROCS="${NPROCS:-52}"
POBS_MODE="${POBS_MODE:-hybrid}"
LIB_VMAG_MAX="${LIB_VMAG_MAX:--6.0}"
RESTART="${RESTART:-0}"
OUTNAME="${OUTNAME:-ngc7793_combined_mid_newread_maxuvfill_priorcache_tangnn_hybrid.h5}"
GALAXY_NAMES=(ngc7793-e ngc7793-w)
LOG_LABEL="ngc7793_combined"

if [[ "${POBS_MODE}" != "observed-box" && "${POBS_MODE}" != "nn" && "${POBS_MODE}" != "hybrid" ]]; then
  echo "ERROR: POBS_MODE must be observed-box, nn, or hybrid, got ${POBS_MODE}" >&2
  exit 1
fi

RESTART_FLAG=()
case "${RESTART}" in
  1|true|TRUE|yes|YES)
    RESTART_FLAG=(--restart)
    ;;
  0|false|FALSE|no|NO)
    ;;
  *)
    echo "ERROR: RESTART must be 0/1, true/false, or yes/no; got ${RESTART}" >&2
    exit 1
    ;;
esac

SAFE_OUTNAME="${OUTNAME%.h5}"
LOG_FILE="${LOG_DIR}/${SAFE_OUTNAME}_${LOG_LABEL}_mid.log"

cd "${BUNDLE_DIR}"

module purge
module load python3/3.11.7
module load hdf5/1.10.5

if [[ ! -f "${VENV}/bin/activate" ]]; then
  echo "ERROR: venv missing at ${VENV}" >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${VENV}/bin/activate"

mkdir -p "${OUT_CHAIN_DIR}" "${LOG_DIR}"

{
  echo "RUN=ngc7793_combined MID new reader no photerr>0.3 cut + max-mag missing UV/U fill + prior-cache weights"
  echo "SCRIPT=${BUNDLE_DIR}/analyze_mid_mdd_new_read.py"
  echo "GALAXY_NAMES=${GALAXY_NAMES[*]}"
  echo "CLUSTER_SLUG_LIB_NAME=${CLUSTER_SLUG_LIB_NAME}"
  echo "OUTNAME=${OUTNAME}"
  echo "POBS_MODE=${POBS_MODE}"
  echo "LIB_VMAG_MAX=${LIB_VMAG_MAX}"
  echo "NITER=${NITER}"
  echo "NPROCS=${NPROCS}"
  echo "RESTART=${RESTART}"
  python3 -c "import torch, sklearn; print('torch', torch.__version__, torch.__file__); print('sklearn', sklearn.__version__, sklearn.__file__)"

  python "${BUNDLE_DIR}/analyze_mid_mdd_new_read.py" \
    "${CLUSTER_SLUG_LIB_NAME}" \
    "${CLUSTER_SLUG_LIB_DIR}/lib_mass.pdf" \
    "${CLUSTER_SLUG_LIB_DIR}/lib_time.pdf" \
    "${CLUSTER_SLUG_LIB_DIR}/lib_av.pdf" \
    --galaxy-names "${GALAXY_NAMES[@]}" \
    --catalog-glob "hlsp_legus*{galaxy_name}*.tab" \
    --disable-hybrid-clean-criteria \
    --legus-cct-root "${LEGUS_CCT_ROOT}" \
    --legus-tab-dir "${LEGUS_TAB_DIR}" \
    --cluster-slug-lib-dir "${CLUSTER_SLUG_LIB_DIR}" \
    --nn-dir "${NN_DIR}" \
    --output-mcmc-chains-dir "${OUT_CHAIN_DIR}" \
    --cattype LEGUS \
    --nwalkers 100 \
    --nprocs "${NPROCS}" \
    --niter "${NITER}" \
    --bwphot 0.05 \
    --bwphys 0.05 \
    --hybrid-range-margin 0.05 \
    --pobs-mode "${POBS_MODE}" \
    --lib-vmag-max "${LIB_VMAG_MAX}" \
    --comp-threshold 1e-12 \
    --outname "${OUTNAME}" \
    "${RESTART_FLAG[@]}" \
    --verbose
} 2>&1 | tee "${LOG_FILE}"
