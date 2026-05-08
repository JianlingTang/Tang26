#!/bin/bash
#PBS -P jh2
#PBS -q expresssr
#PBS -l walltime=24:00:00
#PBS -l ncpus=104
#PBS -l mem=100GB
#PBS -l jobfs=10GB
#PBS -l wd
#PBS -N nn_ngc3344_mdd
#PBS -j oe
#PBS -m bea
#PBS -M janet.tang@anu.edu.au
#PBS -o /scratch/jh2/jt4478/output/nn_ngc3344_mdd.o
#PBS -e /scratch/jh2/jt4478/output/nn_ngc3344_mdd.e
#PBS -l storage=scratch/jh2+gdata/jh2

GALAXY="ngc3344"
MODE="mdd"
OUTNAME="ngc3344_mdd_no_hybrid_pobs.h5"
export GALAXY MODE OUTNAME
exec /g/data/jh2/jt4478/Tang26B/pipeline/scripts/nn_mid_mdd_jobs/run_one_nn_mid_mdd.sh
