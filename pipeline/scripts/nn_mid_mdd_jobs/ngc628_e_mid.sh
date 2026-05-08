#!/bin/bash
#PBS -P jh2
#PBS -q normalsr
#PBS -l walltime=48:00:00
#PBS -l ncpus=104
#PBS -l mem=100GB
#PBS -l jobfs=10GB
#PBS -l wd
#PBS -N nn_ngc628_e_mid
#PBS -j oe
#PBS -m bea
#PBS -M janet.tang@anu.edu.au
#PBS -o /scratch/jh2/jt4478/output/nn_ngc628_e_mid.o
#PBS -e /scratch/jh2/jt4478/output/nn_ngc628_e_mid.e
#PBS -l storage=scratch/jh2+gdata/jh2

GALAXY="ngc628-e"
MODE="mid"
OUTNAME="ngc628-e_mid.h5"
export GALAXY MODE OUTNAME
exec /g/data/jh2/jt4478/Tang26B/pipeline/scripts/nn_mid_mdd_jobs/run_one_nn_mid_mdd.sh
