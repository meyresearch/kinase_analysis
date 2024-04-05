# Functions for the objective function in the hyperparameter optimization of the MSM
# Some are used in model validation 

from pyemma.coordinates import tica as pm_tica
from pyemma.coordinates import cluster_kmeans
from deeptime.markov.msm import MaximumLikelihoodMSM

import gc
from typing import *
import numpy as np
from addict import Dict as Adict
from natsort import natsorted
from tqdm import tqdm
import time
import pandas as pd
from pathlib import Path
import itertools


def run_study(hyperparameters, features, ftraj_dir, study_name, save_dir:Path, add_to_exist_study=False):
    save_dir.mkdir(parents=True, exist_ok=True)

    ftrajs_all, mapping = get_data(hyperparameters['trajlen__cutoff'][0], features, ftraj_dir)
    hp_table = write_hps(hyperparameters, save_dir, study_name, add_to_exist_study)

    print('Running study: ', study_name)
    print('No of hp trials: ', len(hp_table))

    for _, hps in tqdm(hp_table.iterrows(), total=len(hp_table)):
        hp_dict = Adict(hps.to_dict())
        print(hp_dict)

        trial_dir = save_dir/f'{hp_dict.hp_id}'
        trial_dir.mkdir(parents=True, exist_ok=True)

        bootstrap_hp_trial(hp_dict, ftrajs_all, study_name, save_dir)

    return None


def write_hps(hyperparameters, save_dir, study_name, add_to_exist_study=False):
    combinations = list(itertools.product(*hyperparameters.values()))
    hp_table = pd.DataFrame(combinations, columns=hyperparameters.keys())

    if add_to_exist_study:
        exist_hp_table = pd.load_hdf(save_dir/f'{study_name}.h5', key='hps')
        hp_table['hp_id'] = exist_hp_table.hp_id.max() + hp_table.index + 1
        new_hp_table = pd.concat([exist_hp_table, hp_table], ignore_index=True)
        new_hp_table.reindex(drop=True, inplace=True)
        new_hp_table.to_hdf(save_dir/f'{study_name}.h5', key='hps', mode='w')
    else:
        hp_table.reset_index(inplace=True)
        hp_table.rename(columns={'index': 'hp_id'}, inplace=True)
        hp_table.to_hdf(save_dir/f'{study_name}.h5', key='hps', mode='w')

    return hp_table


def bootstrap_hp_trial(hp_dict, ftrajs_all, study_name, save_dir:Path):
    start_time = time.time()

    n_boot = hp_dict.n__boot
    fitting_func = _estimate_msm
    rng = np.random.default_rng(hp_dict.seed)

    ftraj_lens = [x.shape[0] for x in ftrajs_all]

    if n_boot == 1:
        ftraj_ixs = [np.arange(len(ftraj_lens))]
    else:
        ftraj_ixs = [_bootstrap(ftraj_lens, rng) for _ in range(n_boot)]

    for i, ix in tqdm(enumerate(ftraj_ixs), total=n_boot):
        print('\nBootstrap: ', i)
        f_kmeans = save_dir/f'{hp_dict.hp_id}'/f'bs_{i}_kmeans_centers.npy'
        f_tmat = save_dir/f'{hp_dict.hp_id}'/f'bs_{i}_msm_tmat.npy'
        if f_kmeans.is_file() and f_tmat.is_file(): 
            print('Already exist. Continue')
            continue

        ftrajs = [ftrajs_all[i] for i in ix]
        fitting_func(hp_dict, ftrajs, i, study_name, save_dir)

    print('Time elapsed: ', time.time() - start_time)

    return None


def get_data(trajlen_cutoff, features, ftraj_dir) -> Tuple[List[np.ndarray], Dict[int, int]]:
    # Load selected feature trajectories

    ftrajs_to_load = []
    mapping = {}
    old_to_new_mapping = {}

    for i, feature in enumerate(features):
        assert feature in ['dbdist', 'dbdihed', 'aloop', 'ploopdihed', 'achelix', 'rspine'], 'Feature not recognised.'
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

    return ftrajs_to_load, mapping


def _bootstrap(lengths: np.ndarray, rng: np.random.Generator) -> List[np.ndarray]:
    probs = lengths/np.sum(lengths)
    ix = np.arange(len(lengths))
    probs[-1] = 1 - np.sum(probs[0:-1])
    new_ix = rng.choice(ix, size=len(lengths), p=probs, replace=True)

    return new_ix


def _tica(hp_dict: Dict, ftrajs: List[np.ndarray]):
    lag = hp_dict.tica__lag
    stride = hp_dict.tica__stride
    dim = hp_dict.tica__dim

    tica_mod = pm_tica(ftrajs, lag=lag, stride=stride, dim=dim, kinetic_map=True)
    ttrajs = tica_mod.get_output()

    return ttrajs, tica_mod


def _kmeans(hp_dict: Dict, ttrajs: List[np.ndarray], seed: int):

    n_clusters = hp_dict.cluster__k
    stride = hp_dict.cluster__stride
    max_iter = hp_dict.cluster__maxiter

    kmeans_mod = cluster_kmeans(ttrajs, k=n_clusters, max_iter=max_iter, stride=stride, fixed_seed=seed)
    dtrajs = kmeans_mod.dtrajs

    return dtrajs, kmeans_mod


def _estimate_msm(hp_dict, ftrajs, i, study_name, save_dir):
    print('Estimating MSM: ', i)
    ttrajs, tica_mod = _tica(hp_dict, ftrajs)
    dtrajs, kmeans_mod = _kmeans(hp_dict, ttrajs, hp_dict.seed)
    msm_mod = MaximumLikelihoodMSM(reversible=True).fit_fetch(dtrajs, lagtime=hp_dict.markov__lag)

    print('Saving results')
    np.save(kmeans_mod.clustercenters, save_dir/f'{hp_dict.hp_id}'/f'bs_{i}_kmeans_centers.npy')
    np.save(msm_mod.transition_matrix, save_dir/f'{hp_dict.hp_id}'/f'bs_{i}_msm_tmat.npy')

    result = pd.DataFrame(hp_dict, index=['0'])
    result['bs'] = i
    for i in range(20):
        result[f't{i+2}'] = msm_mod.timescales()[i]
        #result[f'vamp2_{i+2}'] = msm_mod.score(dtrajs, r=i+2)
        #result[f'vamp2eq_{i+2}'] = sum(msm_mod.eigenvalues(i+2)**2)
        result[f'gap_{i+2}'] = msm_mod.timescales()[i]/msm_mod.timescales()[i+1]
    result.to_hdf(save_dir/f'{study_name}.h5', key=f'result_raw', mode='a', format='table', append=True, data_columns=True)
    
    del ttrajs, dtrajs
    gc.collect()

    return None