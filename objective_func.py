import deeptime as dt 
from deeptime.decomposition import TICA
from deeptime.clustering import KMeans
from deeptime.markov.msm import MaximumLikelihoodMSM
from deeptime.decomposition import VAMP

import pyemma as pm
from pyemma.coordinates import tica as pm_tica
from pyemma.coordinates import cluster_kmeans

from typing import *
import numpy as np
from addict import Dict as Adict
from natsort import natsorted
from tqdm import tqdm
from pathlib import Path
import time

def objective(trial) -> Tuple[float, float]:
    start_time = time.time()

    n_boot = 20
    seed = 49587
    score_k = 2
    fitting_func = estimate_msm
    markov_lag = 100

    rng = np.random.default_rng(seed)

    hp_dict = define_hpdict(trial)
    ftrajs_all = get_data(hp_dict)

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


def get_data(hp_dict) -> List[np.ndarray]:  
    ftraj_dir = Path('ftraj_egfr')
    ftraj_f = natsorted([ftraj for ftraj in ftraj_dir.rglob('run*-clone?_dunbrack.npy')])
    ftrajs = [np.load(file) for file in ftraj_f if np.load(file).shape[0]>hp_dict.trajlen__cutoff]
    print(f'Loaded {len(ftrajs)} ftrajs.')

    # Convert angles to cos and sin 
    data = [np.concatenate([np.concatenate([np.cos(ftraj[:,3:]), np.sin(ftraj[:,3:])], axis=1), ftraj[:,0:3]], axis=1) for ftraj in ftrajs]
    return data


def bootstrap(lengths: np.ndarray, rng: np.random.Generator) -> List[np.ndarray]:
    probs = lengths/np.sum(lengths)
    ix = np.arange(len(lengths))
    probs[-1] = 1 - np.sum(probs[0:-1])
    new_ix = rng.choice(ix,size=len(lengths), p=probs, replace=True)

    return new_ix


def define_hpdict(trial) -> Dict: 
    hp_dict = Adict()
    hp_dict.trajlen__cutoff = 1000
    hp_dict.tica__lag = trial.suggest_int('tica_lag', 1, 100)
    hp_dict.tica__dim = trial.suggest_int('tica_dim', 2, 15)
    hp_dict.tica__stride = 10
    hp_dict.cluster__k = trial.suggest_int('n_clusters', 100, 1000)
    hp_dict.cluster__maxiter = 1000
    hp_dict.cluster__stride = 100
    assert hp_dict.trajlen__cutoff >= hp_dict.tica__stride
    assert hp_dict.trajlen__cutoff >= hp_dict.cluster__stride
 
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