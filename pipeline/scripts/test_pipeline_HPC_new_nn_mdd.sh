#!/bin/bash
#PBS -P mk27
#PBS -q rsaa
#PBS -l walltime=48:00:00
#PBS -l ncpus=104
#PBS -l mem=100GB
#PBS -l jobfs=10GB
#PBS -l wd
#PBS -N hybrid_newnn_mdd
#PBS -j oe
#PBS -m bea
#PBS -M janet.tang@anu.edu.au
#PBS -o /scratch/jh2/jt4478/output/hybrid_newnn_mdd.o
#PBS -e /scratch/jh2/jt4478/output/hybrid_newnn_mdd.e
#PBS -l storage=scratch/jh2+gdata/jh2

# --- paths (edit PIPE_ROOT / VENV if your layout differs) ---
PIPE_ROOT="/g/data/jh2/jt4478/Tang26B/pipeline"
BUNDLE_DIR="${PIPE_ROOT}/bundled_pipeline"
VENV="${PIPE_ROOT}/.venv"
LOG_DIR="/g/data/jh2/jt4478/Tang26B/output_io"
LOG_FILE="${LOG_DIR}/hybrid_newnn_mdd.log"

set -euo pipefail
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

python3 -c "import torch, sklearn; print('torch', torch.__version__, torch.__file__); print('sklearn', sklearn.__version__, sklearn.__file__)"

LEGUS_CCT_ROOT="/g/data/jh2/jt4478/make_LEGUS_CCT"
LEGUS_TAB_DIR="/g/data/jh2/jt4478/Tang26B/cluster_data"
CLUSTER_SLUG_LIB_DIR="/g/data/jh2/jt4478/cluster_slug"
CLUSTER_SLUG_LIB_NAME="/g/data/jh2/jt4478/cluster_slug/tang"
NN_DIR="/g/data/jh2/jt4478/Tang26B/nn_models"
OUT_CHAIN_DIR="/g/data/jh2/jt4478/Tang26B/output_chains"

mkdir -p "${OUT_CHAIN_DIR}" "${LOG_DIR}"

python "${BUNDLE_DIR}/analyze_catalog_mid_mdd.py" \
  "${CLUSTER_SLUG_LIB_NAME}" \
  "${CLUSTER_SLUG_LIB_DIR}/lib_mass.pdf" \
  "${CLUSTER_SLUG_LIB_DIR}/lib_time.pdf" \
  "${CLUSTER_SLUG_LIB_DIR}/lib_av.pdf" \
  --galaxy-names ngc628-c ngc628-e \
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
  --hybrid-range-margin 0.5 \
  --mdd \
  --outname "ngc628_new_nn_mdd.h5" \
  --verbose 2>&1 | tee "${LOG_FILE}"
