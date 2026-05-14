#!/bin/bash
#PBS -P mk27
#PBS -q rsaa
#PBS -l walltime=24:00:00
#PBS -l ncpus=104
#PBS -l mem=100GB
#PBS -l jobfs=10GB
#PBS -l wd
#PBS -N nn_ngc628_mid_pobs_hybrid
#PBS -j oe
#PBS -m bea
#PBS -M janet.tang@anu.edu.au
#PBS -o /scratch/jh2/jt4478/output/nn_ngc628_mid_pobs_hybrid_mv6.o
#PBS -e /scratch/jh2/jt4478/output/nn_ngc628_mid_pobs_hybrid_mv6.e
#PBS -l storage=scratch/jh2+gdata/jh2

GALAXY_NAMES="ngc628-c ngc628-e"
MODE="mid"
POBS_MODE="hybrid"
LIB_VMAG_MAX="-6.0"
OUTNAME="ngc628_mid_pobs_hybrid_mv6_padova_test.h5"
export GALAXY_NAMES MODE POBS_MODE LIB_VMAG_MAX OUTNAME
exec /g/data/jh2/jt4478/Tang26B/pipeline/scripts/nn_mid_mdd_jobs/run_one_nn_mid_mdd.sh
