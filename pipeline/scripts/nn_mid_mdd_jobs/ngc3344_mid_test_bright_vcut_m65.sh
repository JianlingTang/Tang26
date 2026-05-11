#!/bin/bash
#PBS -P mk27
#PBS -q rsaa
#PBS -l walltime=24:00:00
#PBS -l ncpus=104
#PBS -l mem=100GB
#PBS -l jobfs=10GB
#PBS -l wd
#PBS -N nn_ngc3344_mid_vcut_m65
#PBS -j oe
#PBS -m bea
#PBS -M janet.tang@anu.edu.au
#PBS -o /scratch/jh2/jt4478/output/nn_ngc3344_mid_test_bright_vcut_m65.o
#PBS -e /scratch/jh2/jt4478/output/nn_ngc3344_mid_test_bright_vcut_m65.e
#PBS -l storage=scratch/jh2+gdata/jh2

GALAXY="ngc3344"
MODE="mid"
POBS_MODE="observed-box"
TEST_POBS_MODE="bright-v-cut"
TEST_V_ABS_CUT="-6.5"
OUTNAME="ngc3344_mid_test_bright_vcut_m65.h5"
export GALAXY MODE POBS_MODE TEST_POBS_MODE TEST_V_ABS_CUT OUTNAME
exec /g/data/jh2/jt4478/Tang26B/pipeline/scripts/nn_mid_mdd_jobs/run_one_nn_mid_mdd_test_pobs.sh
