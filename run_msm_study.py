# Run hyperparameter optimisation/sampling
# Use run_create_study.py to create a storage and a study first 

import pandas as pd
import optuna
from pathlib import Path
from functools import partial
from funcs_objective import *

storage_name = "sqlite:///data/dunbrack_msm.db"
study_name = "markovlag_100ns"
key = 'random_trials'
ftraj_dir = Path('ftraj_egfr')

study = optuna.load_study(study_name=study_name, storage=storage_name)
print(len(study.trials))

objective = partial(objective, study_name=study_name, trial_key=key, markov_lag=100, ftraj_dir=ftraj_dir)
study.optimize(objective, n_trials=100, catch=(ValueError,))
print('Sampling finished.')
print(len(study.trials))

study.trials_dataframe().to_hdf(f'data_egfr/{study_name}.h5', key=key)