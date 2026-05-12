#!/bin/bash
#PBS -P jh2
#PBS -q expresssr
#PBS -l walltime=24:00:00
#PBS -l ncpus=104
#PBS -l mem=100GB
#PBS -l jobfs=10GB
#PBS -l wd
#PBS -N nn_ngc628_mdd_obsbox_r8000
#PBS -j oe
#PBS -m bea
#PBS -M janet.tang@anu.edu.au
#PBS -o /scratch/jh2/jt4478/output/nn_ngc628_mdd_pobs_observed_box_restart_8000.o
#PBS -e /scratch/jh2/jt4478/output/nn_ngc628_mdd_pobs_observed_box_restart_8000.e
#PBS -l storage=scratch/jh2+gdata/jh2

GALAXY_NAMES="ngc628-c ngc628-e"
MODE="mdd"
POBS_MODE="observed-box"
OUTNAME="ngc628_mdd_pobs_observed_box.h5"
RESTART="1"
NITER="12000"
export GALAXY_NAMES MODE POBS_MODE OUTNAME RESTART NITER
exec /g/data/jh2/jt4478/Tang26B/pipeline/scripts/nn_mid_mdd_jobs/run_one_nn_mid_mdd.sh
