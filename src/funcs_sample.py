### Functions to sample frames according to feature values or microstate distributions

from deeptime.markov.sample import *
import mdtraj as md
from typing import *
from functools import reduce
import numpy as np

def sample_frames_by_features(ftrajs_list: List[List[np.ndarray]], ftraj_range_list: List[List[Tuple[float,float]]], n_samples: int, mapping=None) -> np.ndarray:
    """
    Sample frames from the feature trajectories according to the feature ranges

    Parameters
    ----------
    ftrajs_list: list of list of ndarray( (n_i, m) )
        Featurised trajectories
    ftraj_range_list: list of list of tuple( (2) )
        The lower and upper boundary of the corresponding features to sample from
    n_samples: int
        The number of samples to be taken from the 
    mapping: dict
        The mapping from the filtered traj indices to the original indices

    Returns
    -------
    samples: ndarray( (n_samples, 2) )
        For each sample, the trajectory index and frame index.    
    """
    
    assert len(ftrajs_list)==len(ftraj_range_list), 'The number of features and their ranges do not match'
    
    masks = []
    for ftrajs, limits in zip(ftrajs_list, ftraj_range_list):
        masks.append([np.logical_and(ftraj>limits[0], ftraj<limits[1]) for ftraj in ftrajs])
    
    combined_mask = [reduce(np.logical_and, masked_ftrajs) for masked_ftrajs in zip(*masks)]

    frames_to_sample_from = []
    for i, array in enumerate(combined_mask):
        true_indices = np.where(array)[0]
        frames_to_sample_from.extend([[i, j] for j in true_indices])
    samples = np.array(frames_to_sample_from)[np.random.choice(range(len(frames_to_sample_from)), n_samples)]

    if mapping is not None: samples = map_to_original_trajs(samples, mapping)

    return samples


def sample_states_by_distribution(microstate_distribution, connected_states, n_samples) -> Dict[int, int]:
    """
    Decide how many sample to take from states according to the microstate distribution

    Parameters
    ----------
    microstate_distribution: ndarray( (n) )
        A distribution over microstates to sample from
    n_samples: int
        The number of samples to be taken
    
    Returns
    -------
    state_samples_count: dict
        The states to sampled from : the number of samples to be taken
    """

    state_indices = np.random.choice(len(microstate_distribution), size=n_samples, p=microstate_distribution)
    counts = np.bincount(state_indices)
    state_samples_count = {connected_states[i]:count for i, count in enumerate(counts) if count!=0}

    return state_samples_count


def sample_frames_by_states(state_samples_count, dtrajs, mapping=None) -> List[np.ndarray]:
    """
    Sample frames from the states according to the state_samples_count

    Parameters
    ----------
    state_samples_count: dict
        The states to sampled from : the number of samples to be taken
    dtrajs: list of ndarray( (n_i) )
        Discretised trajectories of states
    mapping: dict
        The mapping from the filtered indices to the original indices

    Returns
    -------
    samples: ndarray( (n_samples, 2) )
        For each sample, the trajectory index and frame index. 
    """

    index_states = compute_index_states(dtrajs)
    samples = []
    for state_to_sample_from, n_samples in state_samples_count.items():
        samples.append(np.array(index_states[state_to_sample_from])[np.random.choice(range(len(index_states[state_to_sample_from])), n_samples)])
    samples = np.concatenate(samples)
    if mapping is not None: samples = map_to_original_trajs(samples, mapping)

    return samples


def simulated_traj_to_samples(traj_of_states, dtrajs, stride=1, mapping=None) -> np.ndarray:

    index_states = compute_index_states(dtrajs)
    samples = []
    for state in traj_of_states:
        samples.append(np.array(index_states[state])[np.random.choice(range(len(index_states[state])), 1)])
    samples = np.concatenate(samples)
    if mapping is not None: samples = map_to_original_trajs(samples, mapping, stride=stride)

    return samples


def save_samples(samples, traj_files, save_dir, reference=None):
    """
    Combine and save sampled frames to disk
    """

    frames = [] 

    if len(frames)>1:
        sampled_frames = md.join(frames)
    else:
        sampled_frames = frames[0]
    if reference is not None:
        sampled_frames = sampled_frames.superpose(reference)
    sampled_frames.save(save_dir)

    return None


def map_to_original_trajs(samples, traj_mapping, stride=1) -> List[np.ndarray]:
    """
    Map the sampled filtered trajectory indices to the original trajectory indices
    """
    samples_mapped = [(traj_mapping[sample[0]], sample[1]*stride) for sample in samples]
    return np.array(samples_mapped)