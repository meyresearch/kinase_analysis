#!/bin/sh
#$ -N met_bayesian-af2-merged_markov_lags
#$ -cwd
#$ -l h_rt=00:29:59
#$ -pe sharedmem 32
#$ -l h_vmem=40G
#$ -m beas
#$ -M ryan.zhu@ed.ac.uk

# Initialise the environmental modules
. /etc/profile.d/modules.sh

# Load anaconda
module load anaconda 

# With configuration file edited just activate the environment
conda activate msm
python kinase_analysis/src/run_msm.py \
	--protein met \
	--hps_csv /exports/eddie/scratch/s2135271/kinase_analysis/data/met/met_hps.csv \
	--hp_idx 1 2 \
	--wk_dir /exports/eddie/scratch/s2135271/kinase_analysis/data/met/msm/af2-merged-markov_lags \
	--datasets "met-af2-1ns,.,/exports/eddie/scratch/s2135271/kinase_analysis/data/met/met-af2-1ns/ftrajs/,1" \
		   "met-af2-50ps,.,/exports/eddie/scratch/s2135271/kinase_analysis/data/met/met-af2-50ps/ftrajs/,0.05"
