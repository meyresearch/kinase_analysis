# Functions for the validation and sampling from MSMs

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
from functools import partial
import numpy as np
from addict import Dict as Adict
from natsort import natsorted
from tqdm import tqdm
from pathlib import Path


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