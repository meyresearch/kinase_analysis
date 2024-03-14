# Run hyperparameter optimisation/sampling
# Use run_create_study.py to create a storage and a study first 

import pandas as pd
import optuna
from pathlib import Path
from functools import partial
from funcs_objective import *
import builtins

'''
original_print = builtins.print
def print(*args, **kwargs):
    kwargs.setdefault('flush', True)  # Set flush to True unless it's already specified
    original_print(*args, **kwargs)
builtins.print = print
'''

storage_name = "sqlite:///data_egfr/msm/allfeature_studies/allfeature.db"
study_name = "lag500"
trial_key = 'random_trials'
markov_lag= 500
trajlen__cutoff = 1000
features = ['dbdist', 'dbdihed', 'ploop', 'aloop', 'achelix', 'rspine']
ftraj_dir = Path('data_egfr/ftrajs')

ftrajs_all, _ = get_data(trajlen__cutoff, features, ftraj_dir)
study = optuna.load_study(study_name=study_name, storage=storage_name)
print(len(study.trials))

objective = partial(objective, study_name=study_name, trial_key=trial_key, markov_lag=markov_lag, ftrajs_all=ftrajs_all, cutoff=trajlen__cutoff)
study.optimize(objective, n_trials=100, catch=(ValueError,))
print('Sampling finished.')
print(len(study.trials))

study.trials_dataframe().to_hdf(f'data_egfr/msm/allfeature_studies/{study_name}.h5', key=trial_key, mode='a')