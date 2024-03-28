from typing import *
from pathlib import Path

from funcs_build_msm import *

hyperparameters = {'trajlen__cutoff': [1000],
                    'n__boot': [10], 
                    'seed': [49587],
                    'tica__lag': [10],
                    'tica__stride': [1000], 
                    'tica__dim': [20], 
                    'cluster__k': [1000, 2500, 5000, 7500, 10000, 15000], 
                    'cluster__stride': [1000], 
                    'cluster__maxiter': [1000], 
                    'markov__lag': [100]}
features = ['dbdist', 'dbdihed', 'aloop', 'ploop', 'achelix', 'rspine']
ftraj_dir = Path('data_egfr/ftrajs')
study_name = 'how_many_clusters'
save_dir = Path('data_egfr/msm/how_many_clusters/')

run_study(hyperparameters, features, ftraj_dir, study_name, save_dir, add_to_exist_study=False)