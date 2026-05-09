#!/bin/bash
#PBS -P mk27
#PBS -q rsaa
#PBS -l walltime=24:00:00
#PBS -l ncpus=104
#PBS -l mem=100GB
#PBS -l jobfs=10GB
#PBS -l wd
#PBS -N nn_ngc3344_mid_pobs_hybrid
#PBS -j oe
#PBS -m bea
#PBS -M janet.tang@anu.edu.au
#PBS -o /scratch/jh2/jt4478/output/nn_ngc3344_mid_pobs_hybrid_mv6.o
#PBS -e /scratch/jh2/jt4478/output/nn_ngc3344_mid_pobs_hybrid_mv6.e
#PBS -l storage=scratch/jh2+gdata/jh2

GALAXY="ngc3344"
MODE="mid"
POBS_MODE="hybrid"
LIB_VMAG_MAX="-6.0"
OUTNAME="ngc3344_mid_pobs_hybrid_mv6.h5"
export GALAXY MODE POBS_MODE LIB_VMAG_MAX OUTNAME
exec /g/data/jh2/jt4478/Tang26B/pipeline/scripts/nn_mid_mdd_jobs/run_one_nn_mid_mdd.sh
