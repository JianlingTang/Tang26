
#!/bin/bash
#PBS -P jh2
#PBS -q expresssr
#PBS -l walltime=01:00:00
#PBS -l ncpus=104
#PBS -l mem=100GB
#PBS -l jobfs=10GB
#PBS -l wd
#PBS -N newlib_smoke_test
#PBS -j oe
#PBS -m bea
#PBS -M janet.tang@anu.edu.au
#PBS -o /scratch/jh2/jt4478/output/test_newlib.o
#PBS -e /scratch/jh2/jt4478/output/test_newlib.e
# Filesystems this job reads/writes (drop +scratch/mk27 if you do not use mk27).
#PBS -l storage=scratch/jh2+gdata/jh2

# --- paths (edit PIPE_ROOT / VENV if your layout differs) ---
PIPE_ROOT="/g/data/jh2/jt4478/Tang26B/pipeline"
BUNDLE_DIR="${PIPE_ROOT}/bundled_pipeline"
VENV="${PIPE_ROOT}/.venv"
LOG_DIR="/g/data/jh2/jt4478/Tang26B/output_io"
LOG_FILE="${LOG_DIR}/test.log"

set -euo pipefail
cd "${BUNDLE_DIR}"

# module purge
module load python3/3.11.7
# If 3.11.7 is unavailable on the node, use: module load python3/3.11.0
# module load hdf5/1.10.5

# if [[ ! -f "${VENV}/bin/activate" ]]; then
#   echo "ERROR: venv missing at ${VENV}" >&2
#   echo "On login once: module load python3/3.11.7 && cd ${PIPE_ROOT} && python3 -m venv .venv && source .venv/bin/activate" >&2
#   echo "  pip install -U pip && pip install torch --index-url https://download.pytorch.org/whl/cpu" >&2
#   echo "  pip install \"scikit-learn>=1.6.1\" && pip install numpy scipy astropy emcee numexpr h5py matplotlib joblib slugpy" >&2
#   exit 1
# fi
# # shellcheck source=/dev/null
# source "${VENV}/bin/activate"

# module purge
# module load python3/3.11.7
source /g/data/jh2/jt4478/Tang26B/pipeline/.venv/bin/activate

export HYBRID_NN_SCALER=/g/data/jh2/jt4478/Tang26B/nn_models/scaler_phot_ngc628-c.pkl
export HYBRID_NN_MODEL=/g/data/jh2/jt4478/Tang26B/nn_models/best_model_phot_ngc628-c.pt
export HYBRID_SLUG_LIB=/g/data/jh2/jt4478/cluster_slug
export HYBRID_LEGUS_TAB=/g/data/jh2/jt4478/Tang26B/cluster_data/hlsp_legus_hst_acs-wfc3_ngc628-c_multiband_v1_padagb-mwext-avgapcor.tab
export HYBRID_NN_GALAXY=ngc628-c
export HYBRID_SAMPLE_N=9950000

cd /g/data/jh2/jt4478/Tang26B/pipeline
# python3 -m pytest tests/test_hybrid_comp.py -v --tb=short --basetemp=/g/data/jh2/jt4478/Tang26B/output_io/hybrid_pytest_basetemp
python -m pytest tests/test_hybrid_comp.py -k plus1mag -s -v --tb=short --basetemp=/g/data/jh2/jt4478/Tang26B/output_io/hybrid_libcomp_pytest_basetemp > "${LOG_FILE}" 2>&1