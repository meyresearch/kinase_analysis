import deeptime as dt 
from deeptime.decomposition import TICA
from deeptime.clustering import KMeans
from deeptime.markov.msm import MaximumLikelihoodMSM, BayesianMSM
from deeptime.decomposition import VAMP
from deeptime.util.validation import implied_timescales

import pyemma as pm
from pyemma.coordinates import tica as pm_tica
from pyemma.coordinates import cluster_kmeans

import mdtraj as md
from typing import *
import numpy as np
from addict import Dict as Adict
from natsort import natsorted
from tqdm import tqdm
from pathlib import Path
import time
import h5py
import pandas as pd

ftraj_dir = Path('ftraj_egfr')
ftraj_files = natsorted([ftraj for ftraj in ftraj_dir.rglob('run*-clone?_dunbrack.npy')])

def objective(trial) -> Tuple[float, float]:
    start_time = time.time()

    n_boot = 20
    seed = 49587
    score_k = 2
    fitting_func = estimate_msm
    markov_lag = 100

    rng = np.random.default_rng(seed)

    hp_dict = define_hpdict(trial)
    ftrajs_all, _ = get_data(hp_dict)

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


def get_data(hp_dict) -> (List[np.ndarray], Dict[int, int]):
    ftraj_dir = Path('ftraj_egfr')
    ftraj_files = natsorted([ftraj for ftraj in ftraj_dir.rglob('run*-clone?_dunbrack.npy')])

    ftrajs = []
    traj_mapping = {}
    for file in ftraj_files:
        ftraj = np.load(file)
        if ftraj.shape[0] > hp_dict.trajlen__cutoff:
            ftrajs.append(ftraj)
            traj_mapping[len(ftrajs)-1] = ftraj_files.index(file)
    print(f'Loaded {len(ftrajs)} ftrajs.')

    # Convert angles to cos and sin 
    data = [np.concatenate([np.concatenate([np.cos(ftraj[:,3:]), np.sin(ftraj[:,3:])], axis=1), ftraj[:,0:3]], axis=1) for ftraj in ftrajs]
    return data, traj_mapping


def bootstrap(lengths: np.ndarray, rng: np.random.Generator) -> List[np.ndarray]:
    probs = lengths/np.sum(lengths)
    ix = np.arange(len(lengths))
    probs[-1] = 1 - np.sum(probs[0:-1])
    new_ix = rng.choice(ix,size=len(lengths), p=probs, replace=True)

    return new_ix


def define_hpdict(trial) -> Dict: 
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

    study_name = 'markovlag_100ns'
    hp_df = pd.DataFrame(hp_dict, index=[trial.number])
    with h5py.File(f'data/{study_name}.h5', 'r') as file:
        if f'{study_name}_hps' in file:
            hp_all = pd.read_hdf(f'data/{study_name}.h5', key=f'{search_name}_hps')
            hp_df = pd.concat([hp_all, hp_df], ignore_index=True)
    hp_df.to_hdf(f'data/{study_name}.h5', key=f'{search_name}_hps')

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


def its_convergence(dtrajs: List[np.ndarray], lagtimes=[1,10,50,100,200,500,1000], n_samples=10):
    models = []
    for lagtime in tqdm(lagtimes, total=len(lagtimes)):
        models.append(BayesianMSM(n_samples, lagtime=lagtime).fit(dtrajs).fit_fetch(dtrajs))
    its_data = implied_timescales(models)

    return its_data


def sample_states_by_distribution(microstate_distribution, n_samples) -> List[np.ndarray]:
    state_indices = np.random.choice(len(microstate_distribution), size=n_samples, p=microstate_distribution)
    counts = np.bincount(state_indices)
    state_samples_count = {i:count for i, count in enumerate(counts) if count!=0}

    return state_samples_count


def save_sampled_conf(state_samples_count, frame_of_states, traj_mapping, ftraj_files, traj_dir, save_dir):
    """
    state_samples_count: dict
        The number of samples to be picked from each state
    frame_of_states: dict
        The indices of the states; use compte_index_states
    traj_mapping: dict
        The mapping of the filtered ftraj indices to the original ftraj indices 
    ftraj_files: list
        The list of the ftraj file names
    traj_dir: Path
        The directory of the trajectory files
    save_dir: Path
        The directory to save the sampled frames
    """

    # Pick the samples from the states
    state_samples_idx = {}
    for state, no in state_samples_count.items():
        indicies = np.random.randint(len(frame_of_states[state]), size=no)
        state_samples_idx[state] = [frame_of_states[state][id,:] for id in indicies]

    # Remove the state indicies
    samples = []
    for sample in state_samples_idx.values():
        samples.extend(sample)

    # Map the filtered ftrajs indices to original traj file names
    trajs_frames = {}
    for sample in samples:
        ftraj_idx, t = sample[0], sample[1]
        ftraj_name = ftraj_files[traj_mapping[ftraj_idx]]
        traj_name = traj_dir.joinpath(ftraj_name.stem.split('_')[0]+'.h5')
        if traj_name in trajs_frames:
            trajs_frames[traj_name].append(t)
        else:
            trajs_frames[traj_name] = [t]
    
    # Load, slice, concatenate, and save the frames
    print('Loading trajectories')
    frames = [] 
    for traj_name, ind in tqdm(trajs_frames.items(), total=len(trajs_frames)):
        frames.append(md.load(traj_name)[ind])
    if len(frames)>1:
        sampled_frames = md.join(frames)
    sampled_frames.save(save_dir)

    return frames