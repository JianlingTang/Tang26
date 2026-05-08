#!/bin/bash
#PBS -P jh2
#PBS -q expresssr
#PBS -l walltime=01:00:00
#PBS -l ncpus=104
#PBS -l mem=100GB
#PBS -l jobfs=10GB
#PBS -l wd
#PBS -N test_pipeline_HPC
#PBS -j oe
#PBS -m bea
#PBS -M janet.tang@anu.edu.au
#PBS -o /scratch/jh2/jt4478/output/test_smoke.o
#PBS -e /scratch/jh2/jt4478/output/test_smoke.e
# Filesystems this job reads/writes (drop +scratch/mk27 if you do not use mk27).
#PBS -l storage=scratch/jh2+gdata/jh2

# --- paths (edit PIPE_ROOT / VENV if your layout differs) ---
PIPE_ROOT="/g/data/jh2/jt4478/Tang26B/pipeline"
BUNDLE_DIR="${PIPE_ROOT}/bundled_pipeline"
VENV="${PIPE_ROOT}/.venv"
LOG_DIR="/g/data/jh2/jt4478/Tang26B/output_io"
LOG_FILE="${LOG_DIR}/smoke_test_midmdd.log"

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

python "/g/data/jh2/jt4478/Tang26B/pipeline/tests/test_nn_inputs.py" \
  /g/data/jh2/jt4478/cluster_slug/tang \
  --filters WFC3_UVIS_F275W WFC3_UVIS_F336W ACS_F435W ACS_F555W ACS_F814W \
  --nn-scaler /g/data/jh2/jt4478/Tang26B/nn_models/ngc628-c_scaler.pkl \
  --print-rows 10 \
  --distance-mpc 9.9 > "${LOG_FILE}" 2>&1