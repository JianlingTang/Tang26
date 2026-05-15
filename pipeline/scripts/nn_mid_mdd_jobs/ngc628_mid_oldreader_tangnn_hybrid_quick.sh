#!/bin/bash
#PBS -P jh2
#PBS -q expresssr
#PBS -l walltime=24:00:00
#PBS -l ncpus=104
#PBS -l mem=500GB
#PBS -l jobfs=10GB
#PBS -l wd
#PBS -N nn_ngc628_oldreader
#PBS -j oe
#PBS -m bea
#PBS -M janet.tang@anu.edu.au
#PBS -o /scratch/jh2/jt4478/output/nn_ngc628_mid_oldreader_tangnn_hybrid_quick.o
#PBS -e /scratch/jh2/jt4478/output/nn_ngc628_mid_oldreader_tangnn_hybrid_quick.e
#PBS -l storage=scratch/jh2+gdata/jh2

set -euo pipefail

PIPE_ROOT="/g/data/jh2/jt4478/Tang26B/pipeline"
BUNDLE_DIR="${PIPE_ROOT}/bundled_pipeline"
VENV="${PIPE_ROOT}/.venv"
ROOT="/g/data/jh2/jt4478/Tang26B"
LOG_DIR="${ROOT}/output_io"
OUT_CHAIN_DIR="${ROOT}/output_chains"

CLUSTER_SLUG_LIB_DIR="${CLUSTER_SLUG_LIB_DIR:-/g/data/jh2/jt4478/cluster_slug}"
CLUSTER_SLUG_LIB_NAME="${CLUSTER_SLUG_LIB_NAME:-/g/data/jh2/jt4478/cluster_slug/tang_padova}"
NN_DIR="${NN_DIR:-${ROOT}/nn_models}"
NITER="${NITER:-5000}"
NPROCS="${NPROCS:-52}"
POBS_MODE="${POBS_MODE:-hybrid}"
LIB_VMAG_MAX="${LIB_VMAG_MAX:--6.0}"
OUTNAME="${OUTNAME:-ngc628_mid_oldreader_tangnn_${POBS_MODE}_quick.h5}"

OLD_C="/g/data/jh2/jt4478/legus_slug23/cluster_catalogs/hlsp_628c.tab"
OLD_E="/g/data/jh2/jt4478/legus_slug23/cluster_catalogs/hlsp_628e.tab"

if [[ "${POBS_MODE}" != "observed-box" && "${POBS_MODE}" != "nn" && "${POBS_MODE}" != "hybrid" ]]; then
  echo "ERROR: POBS_MODE must be observed-box, nn, or hybrid, got ${POBS_MODE}" >&2
  exit 1
fi

SAFE_OUTNAME="${OUTNAME%.h5}"
LOG_FILE="${LOG_DIR}/${SAFE_OUTNAME}_ngc628_c_ngc628_e_mid.log"

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
  echo "RUN=old reader + Tang26B NN"
  echo "CATALOGS=${OLD_C} ${OLD_E}"
  echo "CLUSTER_SLUG_LIB_NAME=${CLUSTER_SLUG_LIB_NAME}"
  echo "OUTNAME=${OUTNAME}"
  echo "POBS_MODE=${POBS_MODE}"
  echo "LIB_VMAG_MAX=${LIB_VMAG_MAX}"
  echo "NITER=${NITER}"
  echo "NPROCS=${NPROCS}"
  python3 -c "import torch, sklearn; print('torch', torch.__version__, torch.__file__); print('sklearn', sklearn.__version__, sklearn.__file__)"

  python "${BUNDLE_DIR}/analyze_catalog_mid_mdd.py" \
    "${CLUSTER_SLUG_LIB_NAME}" \
    "${CLUSTER_SLUG_LIB_DIR}/lib_mass.pdf" \
    "${CLUSTER_SLUG_LIB_DIR}/lib_time.pdf" \
    "${CLUSTER_SLUG_LIB_DIR}/lib_av.pdf" \
    "${OLD_C}" "${OLD_E}" \
    --old-ngc628-reader \
    --disable-hybrid-clean-criteria \
    --legus-cct-root "/g/data/jh2/jt4478/make_LEGUS_CCT" \
    --legus-tab-dir "${ROOT}/cluster_data" \
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
    --restart \
    --verbose
} 2>&1 | tee "${LOG_FILE}"
