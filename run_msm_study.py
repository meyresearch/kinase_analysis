# Run hyperparameter optimisation/sampling
# Use run_create_study.py to create a storage and a study first 

import pandas as pd
import optuna
from pathlib import Path
from functools import partial
from funcs_objective import *

storage_name = "sqlite:///data_egfr/msm/dunbrack.db"

study_name = "lag100"
trial_key = 'random_trials'
markov_lag= 100
features = ['dbdist', 'dbdihed']
ftraj_dir = Path('data_egfr/ftrajs')

study = optuna.load_study(study_name=study_name, storage=storage_name)
print(len(study.trials))

objective = partial(objective, study_name=study_name, trial_key=trial_key, markov_lag=markov_lag, features=features, ftraj_dir=ftraj_dir)
study.optimize(objective, n_trials=100, catch=(ValueError,))
print('Sampling finished.')
print(len(study.trials))

study.trials_dataframe().to_hdf(f'data_egfr/msm/{study_name}.h5', key=trial_key, mode='a')