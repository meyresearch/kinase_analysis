import pandas as pd
import optuna

from objective_func import *

storage_name = "sqlite:///data/dunbrack_msm.db"
study_name = "markovlag_100ns"

study = optuna.load_study(study_name=study_name, storage=storage_name)
print(len(study.trials))

study.optimize(objective, timeout=12*60*60, catch=(ValueError,))
print('Sampling finished.')
print(len(study.trials))

study.trials_dataframe().to_hdf(f'data/.h5', key=f'{study_name}')