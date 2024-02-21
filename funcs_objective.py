# Functions for the objective function in the hyperparameter optimization of the MSM
# Some are used in model validation 

from pyemma.coordinates import tica as pm_tica
from pyemma.coordinates import cluster_kmeans
from deeptime.markov.msm import MaximumLikelihoodMSM

from typing import *
import numpy as np
from addict import Dict as Adict
from natsort import natsorted
from tqdm import tqdm
import time
import h5py
import pandas as pd


def objective(trial, study_name, trial_key, markov_lag, ftraj_dir) -> Tuple[float, float]:
    start_time = time.time()

    n_boot = 20
    seed = 49587
    score_k = 2
    fitting_func = estimate_msm

    rng = np.random.default_rng(seed)

    hp_dict = define_hpdict(trial, study_name, trial_key)
    ftrajs_all, _ = get_data(hp_dict, ftraj_dir)

    ftraj_lens = [x.shape[0] for x in ftrajs_all]
    ixs = [bootstrap(ftraj_lens, rng) for _ in range(n_boot)]

    results = []
    for ix in tqdm(ixs, total=n_boot):
        results.append(fitting_func(ix, ftrajs_all, hp_dict, seed, markov_lag, score_k))

    res0 = np.median([res[0] for res in results])
    res1 = np.median([res[1] for res in results])
    res2 = np.median([res[2] for res in results])
    res3 = np.median([res[3] for res in results])

    print('Time elapsed: ', time.time()-start_time)

    return res0, res1, res2, res3


def get_data(hp_dict, ftraj_dir) -> (List[np.ndarray], Dict[int, int]):
    ftraj_files = natsorted([ftraj for ftraj in ftraj_dir.rglob('run*-clone?_dunbrack.npy')])

    ftrajs = []
    traj_mapping = {}
    for file in ftraj_files:
        ftraj = np.load(file)
        if ftraj.shape[0] > hp_dict.trajlen__cutoff:
            ftrajs.append(ftraj)
            traj_mapping[len(ftrajs)-1] = ftraj_files.index(file)
    print(f'Loaded {len(ftrajs)} ftrajs.')

    return ftrajs, traj_mapping


def bootstrap(lengths: np.ndarray, rng: np.random.Generator) -> List[np.ndarray]:
    probs = lengths/np.sum(lengths)
    ix = np.arange(len(lengths))
    probs[-1] = 1 - np.sum(probs[0:-1])
    new_ix = rng.choice(ix,size=len(lengths), p=probs, replace=True)

    return new_ix


def define_hpdict(trial, study_name, trial_key) -> Dict: 
    search_name = 'random_trials'

    hp_dict = Adict()
    hp_dict.trial__no = trial.number
    hp_dict.trajlen__cutoff = 1000
    hp_dict.tica__lag = trial.suggest_int('tica__lag', 1, 100)
    hp_dict.tica__dim = trial.suggest_int('tica__dim', 2, 15)
    hp_dict.tica__stride = 10
    hp_dict.cluster__k = trial.suggest_int('cluster__k', 100, 1000)
    hp_dict.cluster__maxiter = 1000
    hp_dict.cluster__stride = 100

    assert hp_dict.trajlen__cutoff >= hp_dict.tica__stride
    assert hp_dict.trajlen__cutoff >= hp_dict.cluster__stride

    hp_df = pd.DataFrame(hp_dict, index=[trial.number])
    with h5py.File(f'data/{study_name}.h5', 'r') as file:
        if f'{study_name}_hps' in file:
            hp_all = pd.read_hdf(f'data/{study_name}.h5', key=f'{search_name}_hps')
            hp_df = pd.concat([hp_all, hp_df], ignore_index=True)
    hp_df.to_hdf(f'data_egfr/{study_name}.h5', key=f'{trial_key}_hps')

    return hp_dict


def tica(hp_dict: Dict, ftrajs: List[np.ndarray]):
    lag = hp_dict.tica__lag
    stride = hp_dict.tica__stride
    dim = hp_dict.tica__dim

    tica_mod = pm_tica(ftrajs, lag=lag, stride=stride, dim=dim, kinetic_map=True)
    ttrajs = tica_mod.get_output()

    return ttrajs, tica_mod


def kmeans(hp_dict: Dict, ttrajs: List[np.ndarray], seed: int):
    n_clusters = hp_dict.cluster__k
    stride = hp_dict.cluster__stride
    max_iter = hp_dict.cluster__maxiter

    kmeans_mod = cluster_kmeans(ttrajs, k=n_clusters, max_iter=max_iter, stride=stride, fixed_seed=seed)
    dtrajs = kmeans_mod.dtrajs

    return dtrajs, kmeans_mod


def estimate_msm(ix, ftrajs_all, hp_dict, seed, markov_lag, score_k):
    ftrajs = [ftrajs_all[i] for i in ix]

    ttrajs, tica_mod = tica(hp_dict, ftrajs)
    dtrajs, kmeans_mod = kmeans(hp_dict, ttrajs, seed)    

    msm_mod = MaximumLikelihoodMSM(reversible=True).fit_fetch(dtrajs, lagtime=markov_lag)
    t2 = msm_mod.timescales()[1]
    t3 = msm_mod.timescales()[2]
    t4 = msm_mod.timescales()[3]
    t2_gap = t2/t3
    t3_gap = t3/t4

    return t2, t3, t2_gap, t3_gap