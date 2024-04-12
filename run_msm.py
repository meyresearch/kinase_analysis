from typing import *
from pathlib import Path
import gc

from funcs_build_msm import *

gc.enable()
hyperparameters = {'trajlen__cutoff': [1000],
                   'n__boot': [20], 
                   'seed': [49587],
                   'tica__lag': [10],
                   'tica__stride': [1000], 
                   'tica__dim': [20], 
                   'cluster__k': [1000, 2500, 5000, 10000], 
                   'cluster__stride': [1000], 
                   'cluster__maxiter': [1000], 
                   'markov__lag': [1]}
features = ['dbdist', 'dbdihed', 'aloop', 'ploopdihed', 'achelix']
ftraj_dir = Path('/exports/eddie/scratch/s2135271/data_egfr/ftrajs')
study_name = 'n_clusters'
save_dir = Path(f'/exports/csce/eddie/chem/groups/Mey/Ryan/kinase_analysis/data_egfr/msm/{study_name}')

run_study(hyperparameters, features, ftraj_dir, study_name, save_dir, add_to_exist_study=False)
