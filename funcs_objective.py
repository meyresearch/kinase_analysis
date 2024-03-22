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
import pandas as pd
from pathlib import Path


def objective(trial, study_name, trial_key, markov_lag, ftrajs_all, cutoff) -> Tuple[float, float, float, float]:
    start_time = time.time()

    n_boot = 10
    seed = 49587
    score_k = 2
    fitting_func = estimate_msm
    rng = np.random.default_rng(seed)

    hp_dict = define_hpdict(trial, study_name, trial_key, cutoff)

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


def get_data(trajlen_cutoff, features, ftraj_dir) -> Tuple[List[np.ndarray], Dict[int, int]]:
    # Load selected feature trajectories

    ftrajs_to_load = []
    mapping = {}
    old_to_new_mapping = {}

    for i, feature in enumerate(features):
        assert feature in ['dbdist', 'dbdihed', 'aloop', 'ploop', 'achelix', 'rspine'], 'Feature not recognised.'
        ftraj_files = natsorted([str(ftraj) for ftraj in ftraj_dir.rglob(f'run*-clone?_{feature}.npy')])
        print("Loading feature: ", feature)

        for old_idx, ftraj_file in tqdm(enumerate(ftraj_files), total=len(ftraj_files)):
            ftraj = np.load(ftraj_file, allow_pickle=True)
            
            # Process feature specific adjustments
            if 'dihed' in feature:
                ftraj = np.concatenate([np.cos(ftraj), np.sin(ftraj)], axis=1)

            # For the first loaded feature, check the trajectory length
            if i == 0:
                if ftraj.shape[0] > trajlen_cutoff:
                    ftrajs_to_load.append(ftraj)
                    mapping[len(ftrajs_to_load)-1] = old_idx
                    old_to_new_mapping[old_idx] = len(ftrajs_to_load)-1
            else:
                # For subsequent features, concatenate with existing trajectories
                # Ensure to match the length criteria before adding
                if ftraj.shape[0] > trajlen_cutoff:
                    existing_ftraj = ftrajs_to_load[old_to_new_mapping[old_idx]]
                    combined_ftraj = np.hstack([existing_ftraj, ftraj])
                    ftrajs_to_load[old_to_new_mapping[old_idx]] = combined_ftraj

    print(f'Loaded number of ftrajs: {len(ftrajs_to_load)}')
    
    '''
    for i, feature in enumerate(features):
        assert feature in ['aloop', 'dbdist', 'dbdihed', 'ploop', 'achelix', 'rspine'], 'Feature not recognised.'
        ftraj_files = natsorted([str(ftraj) for ftraj in ftraj_dir.rglob(f'run*-clone?_{feature}.npy')])
        print(feature)
        if i == 0:
            ftrajs_cat = [np.load(ftraj_file, allow_pickle=True) for ftraj_file in ftraj_files]

            # Remove trajectories with length less than the cutoff
            # Meanwhile create a mapping from the new indices to the old indices
            ftrajs_len = np.array([ftraj.shape[0] for ftraj in ftrajs_cat])
            ftrajs_len_mask = ftrajs_len > trajlen__cutoff
            ftrajs_cat = [ftrajs_cat[i] for i in range(len(ftrajs_cat)) if ftrajs_len_mask[i]]
            mapping = {new_idx: old_idx for new_idx, old_idx in enumerate(np.where(ftrajs_len_mask == 1)[0])}
        else:
            ftraj_files = [ftraj_files[i] for i in range(len(ftraj_files)) if ftrajs_len_mask[i]]
            ftrajs_new = [np.load(ftraj_file, allow_pickle=True) for ftraj_file in ftraj_files]
            if 'dihed' in feature: ftrajs_new = [np.concatenate([np.cos(ftraj), np.sin(ftraj)], axis=1) for ftraj in ftrajs_new]
            ftrajs_cat = [np.hstack([ftraj, ftraj_new]) for ftraj, ftraj_new in zip(ftrajs_cat, ftrajs_new)]

    ftrajs_all = []
    for feature in features:
        assert feature in ['dbdist', 'dbdihed', 'aloop', 'ploop', 'achelix', 'rspine'], 'Feature not recognised.'
        ftraj_files = natsorted([str(ftraj) for ftraj in ftraj_dir.rglob(f'run*-clone?_{feature}.npy')])
        ftrajs = [np.load(ftraj_file, allow_pickle=True).astype(np.float16) for ftraj_file in ftraj_files]
        # Convert dihedral angles to sin and cos pairs
        if 'dihed' in feature: ftrajs = [np.concatenate([np.cos(ftraj), np.sin(ftraj)], axis=1) for ftraj in ftrajs]
        ftrajs_all.append(ftrajs)

    print(f'Total number of ftrajs: {len(ftrajs_all[0])}.')
    ftrajs_cat = [np.concatenate([lst[i] for lst in ftrajs_all], axis=1).astype(np.float32) for i in range(len(ftrajs))]
    del ftrajs_all
    

    ftrajs_len = np.array([ftraj.shape[0] for ftraj in ftrajs_cat])
    ftrajs_len_mask = ftrajs_len > trajlen__cutoff
    ftraj_to_load = [ftrajs_cat[i] for i in range(len(ftrajs_cat)) if ftrajs_len_mask[i]]
    mapping = {new_idx: old_idx for new_idx, old_idx in enumerate(np.where(ftrajs_len_mask == 1)[0])}
    '''

    return ftrajs_to_load, mapping


def bootstrap(lengths: np.ndarray, rng: np.random.Generator) -> List[np.ndarray]:
    probs = lengths/np.sum(lengths)
    ix = np.arange(len(lengths))
    probs[-1] = 1 - np.sum(probs[0:-1])
    new_ix = rng.choice(ix,size=len(lengths), p=probs, replace=True)

    return new_ix


def define_hpdict(trial, study_name, trial_key, cutoff) -> Dict: 
    hp_dict = Adict()
    hp_dict.trial__no = trial.number
    hp_dict.tica__lag = trial.suggest_int('tica__lag', 5, 100)
    hp_dict.tica__dim = trial.suggest_int('tica__dim', 5, 50)
    hp_dict.tica__stride = 10
    hp_dict.trajlen__cutoff = cutoff
    hp_dict.cluster__k = trial.suggest_int('cluster__k', 8000, 12000)
    hp_dict.cluster__maxiter = 1000
    hp_dict.cluster__stride = 100

    assert hp_dict.trajlen__cutoff >= hp_dict.tica__stride
    assert hp_dict.trajlen__cutoff >= hp_dict.cluster__stride

    hp_df = pd.DataFrame(hp_dict, index=[trial.number])
    hp_df.to_hdf(f'data_egfr/msm/allfeature_studies/{study_name}.h5', key=f'{trial_key}_hps', mode='a', format='table', append=True, data_columns=True)

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