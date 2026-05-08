#!/bin/bash
#PBS -P jh2
#PBS -q expresssr
#PBS -l walltime=01:00:00
#PBS -l ncpus=104
#PBS -l mem=100GB
#PBS -l jobfs=10GB
#PBS -l wd
#PBS -N nn_nomcmc_diag
#PBS -j oe
#PBS -m bea
#PBS -M janet.tang@anu.edu.au
#PBS -o /scratch/jh2/jt4478/output/nn_nomcmc_diag.o
#PBS -e /scratch/jh2/jt4478/output/nn_nomcmc_diag.e
#PBS -l storage=scratch/jh2+gdata/jh2

set -euo pipefail

PIPE_ROOT="/g/data/jh2/jt4478/Tang26B/pipeline"
VENV="${PIPE_ROOT}/.venv"
LOG_DIR="/g/data/jh2/jt4478/Tang26B/output_io"
OUT_DIR="${LOG_DIR}/nn_nomcmc_diag_multi"
LOG_FILE="${LOG_DIR}/nn_nomcmc_diag_multi.log"

LEGUS_CCT_ROOT="/g/data/jh2/jt4478/make_LEGUS_CCT"
LEGUS_TAB_DIR="/g/data/jh2/jt4478/Tang26B/cluster_data"
CLUSTER_SLUG_LIB_NAME="/g/data/jh2/jt4478/cluster_slug/tang"
NN_DIR="/g/data/jh2/jt4478/Tang26B/nn_models"

# Set MAX_LIB_ROWS=0 when submitting if you want the full tang library.
MAX_LIB_ROWS="${MAX_LIB_ROWS:-250000}"

mkdir -p "${LOG_DIR}" "${OUT_DIR}"

module purge
module load python3/3.11.7
module load hdf5/1.10.5

if [[ ! -f "${VENV}/bin/activate" ]]; then
  echo "ERROR: venv missing at ${VENV}" >&2
  exit 1
fi
source "${VENV}/bin/activate"

python3 -c "import torch, sklearn; print('torch', torch.__version__, torch.__file__); print('sklearn', sklearn.__version__, sklearn.__file__)"

cd "${PIPE_ROOT}"
python "${PIPE_ROOT}/tests/diagnose_nn_pipeline_no_mcmc.py" \
  "${CLUSTER_SLUG_LIB_NAME}" \
  --galaxy-names ngc628-c ngc628-e ngc1566 ngc7793-e ngc7793-w \
  --catalog-glob "hlsp_legus*{galaxy_name}*.tab" \
  --legus-cct-root "${LEGUS_CCT_ROOT}" \
  --legus-tab-dir "${LEGUS_TAB_DIR}" \
  --nn-dir "${NN_DIR}" \
  --output-dir "${OUT_DIR}" \
  --comp-threshold 0.01 \
  --lib-vmag-max -6.0 \
  --hybrid-range-margin 0.5 \
  --hybrid-min-bands-nonzero 4 \
  --max-lib-rows "${MAX_LIB_ROWS}" \
  --plot-sample 800000 \
  2>&1 | tee "${LOG_FILE}"

echo "Wrote diagnostics to ${OUT_DIR}"
echo "Log: ${LOG_FILE}"
