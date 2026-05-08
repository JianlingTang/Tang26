#!/bin/bash
#PBS -P jh2
#PBS -q normalsr
#PBS -l walltime=48:00:00
#PBS -l ncpus=104
#PBS -l mem=100GB
#PBS -l jobfs=10GB
#PBS -l wd
#PBS -N nn_ngc5457_mdd
#PBS -j oe
#PBS -m bea
#PBS -M janet.tang@anu.edu.au
#PBS -o /scratch/jh2/jt4478/output/nn_ngc5457_mdd.o
#PBS -e /scratch/jh2/jt4478/output/nn_ngc5457_mdd.e
#PBS -l storage=scratch/jh2+gdata/jh2

GALAXY_NAMES="ngc5457-c ngc5457-nw1 ngc5457-nw2 ngc5457-nw3 ngc5457-se"
GALAXY_LABEL="ngc5457"
MODE="mdd"
OUTNAME="ngc5457_combined_mdd.h5"
export GALAXY_NAMES GALAXY_LABEL MODE OUTNAME
exec /g/data/jh2/jt4478/Tang26B/pipeline/scripts/nn_mid_mdd_jobs/run_one_nn_mid_mdd.sh
