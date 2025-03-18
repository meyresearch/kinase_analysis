from deeptime.markov.sample import compute_index_states
import deeptime as dt
import mdtraj as md 

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from natsort import natsorted
from addict import Dict as Adict
from collections import defaultdict


class MSMStudy():
    '''
    This class reconstruct a MSM model from hyperparameters, saved trajectories, and models. 
    '''

    def __init__(self, hps_table, traj_data, wk_dir):
        '''
        Parameters
        ----------
        hps_table : pd.DataFrame
            The hyperparameter table for the MSM estimation
        traj_data : TrajData
            The TrajData object containing the feature trajectories
        wk_dir : str or Path
            The directory contains the saved MSM models, trajectories, and observations
        '''

        self.protein = traj_data.protein
        self.wk_dir = Path(wk_dir)
        self.hps_table = hps_table
        self.observation = pd.read_csv(self.wk_dir / 'observation.csv')
        self.traj_data = traj_data

        self._hp_id = None
        self._pcca_n = None


    def set_hp_id(self, hp_id):
        '''
        Select the model index to load
        '''
        assert hp_id in self.hps_table['hp_id'].values, f'hp_id {hp_id} not found in the hyperparameter table.'
        assert (self.wk_dir / str(hp_id)).exists(), f'Study directory {self.wk_dir / str(hp_id)} not found.'
        self._hp_id = hp_id

        self.save_dir = self.wk_dir / str(hp_id)
        self.fig_dir = self.save_dir / 'figs'
        self.fig_dir.mkdir(exist_ok=True)
        self.sample_dir = self.save_dir / 'samples'
        self.sample_dir.mkdir(exist_ok=True)     
        self.hp_dict = Adict(self.hps_table.loc[self.hps_table['hp_id'] == hp_id].squeeze().to_dict())

        for key in self.traj_data.datasets.keys():
            ftraj_dt = self.traj_data.datasets[key]['dt']
            dt_out = self.hp_dict['dt_out']
            self.traj_data.datasets[key]['stride'] = int(dt_out/ftraj_dt)
            print(f'Set dataset {key} stride to {self.traj_data.datasets[key]["stride"]}')

        print(f'Loading MSM model id {hp_id} from {self.save_dir}')
        print(f'{self.hp_dict}')

        self.load_all()


    def load_all(self):

        print('Loading trajectories...')
        ttraj_dir = self.save_dir/'ttrajs'
        ttraj_files = natsorted(list(ttraj_dir.glob('*.npy')))
        self.ttrajs = [np.load(f) for f in ttraj_files]
        self.ttraj_cat = np.concatenate(self.ttrajs, axis=0)

        dtraj_dir = self.save_dir/'dtrajs'
        dtraj_files = natsorted(list(dtraj_dir.glob('*.npy')))
        self.dtrajs = [np.load(f) for f in dtraj_files]
        self.dtraj_cat = np.concatenate(self.dtrajs, axis=0)   

        print('Loading models...')
        model_dir = self.save_dir/'models'
        with open(model_dir/'mapping.json', 'rb') as f:
            self.mapping = {int(k):int(v) for k,v in json.load(f).items()}
        with open(model_dir/'tica_model.pkl', 'rb') as f:
            self.tica_mod = pickle.load(f)
        with open(model_dir/'kmeans_model.pkl', 'rb') as f:
            self.kmeans_mod = pickle.load(f)
        self.kmeans_centers = self.kmeans_mod.clustercenters
        with open(model_dir/'count_model.pkl', 'rb') as f:
            self.count_mod = pickle.load(f)
        if self.hp_dict.msm_mode == 'maximum_likelihood':
            with open(model_dir/'maximum_likelihood_msm_model.pkl', 'rb') as f:
                self.msm_mod = pickle.load(f)
        elif self.hp_dict.msm_mode == 'bayesian':
            with open(model_dir/'bayesian_msm_model.pkl', 'rb') as f:
                self.baymsm_mod = pickle.load(f)
            self.msm_mod = self.baymsm_mod.prior
        self.traj_weights = self.msm_mod.compute_trajectory_weights(self.dtrajs)
        self.connected_states = np.load(model_dir/'connected_states.npy')
        self.disconnected_states = np.setdiff1d(np.arange(self.hp_dict.cluster_n), self.connected_states) 

        print('Done')


    def run_pcca(self, n):
        '''
        Run PCCA+ on the MSM model
        '''
        self._pcca_n = n
        self.pcca_mod = self.msm_mod.pcca(n)
        self.pcca_assignment = self.pcca_mod.assignments
        self.micro_to_macro = {self.connected_states[i]: self.pcca_assignment[i] for i in range(len(self.connected_states))}
        self.ptraj_cat = np.array([self.micro_to_macro[d] if d in self.connected_states else -1 for d in self.dtraj_cat])


    def transform(self, ftraj):
        if not self.tica_mod:
            raise ValueError('TICA model not found. Run estimate_MSM or restore_models first.')
        if not self.kmeans_mod:
            raise ValueError('Kmeans model not found. Run estimate_MSM or restore_models first.')
        if not self.msm_mod:
            raise ValueError('MSM model not found. Run estimate_MSM or restore_models first.')
        if not self.pcca_mod:
            raise ValueError('PCCA model not found. Run estimate_MSM or restore_models first.')
        
        ttraj = self.tica_mod.transform(ftraj)
        dtraj = self.kmeans_mod.transform(ttraj)[0].flatten()
        connected_d = np.array([idx for idx in dtraj if idx in self.connected_states]).flatten()
        disconnected_d = np.array([idx for idx in dtraj if idx in self.disconnected_states]).flatten()
        pcca_assignment = np.array([self.micro_to_macro[d] for d in connected_d])

        return ttraj, dtraj, connected_d, disconnected_d, pcca_assignment
    

    def _get_state_count_from_distrib(self, microstate_weights, connected_states, n_samples):
        """
        Decide how many sample to take from states according to microstate weights

        Parameters
        ----------
        microstate_distribution: ndarray( (n) )
            A distribution over microstates to sample from
        n_samples: int
            The number of samples to be taken
        
        Returns
        -------
        state_samples_count: dict[int, int]
            The states to sampled from : the number of samples to be taken
        """
        
        assert len(microstate_weights) == len(connected_states), 'The distribution should have the same length as the connected states'

        state_indices = np.random.choice(len(microstate_weights), size=n_samples, p=microstate_weights)
        counts = np.bincount(state_indices)
        state_samples_count = {connected_states[i]:count for i, count in enumerate(counts) if count!=0}

        return state_samples_count


    def _get_samples_from_state(self, state_samples_count):
        index_states = compute_index_states(self.dtrajs)
        samples = []
        for state_to_sample_from, n_samples in state_samples_count.items():
            samples.append(np.array(index_states[state_to_sample_from])[np.random.choice(range(len(index_states[state_to_sample_from])), n_samples)])
        samples = np.concatenate(samples)

        return samples
    

    def sample_from_distrib(self, distrib):
        state_samples_count = self._get_state_count_from_distrib(distrib, self.connected_states, len(distrib))
        samples = self._get_samples_from_state(state_samples_count)

        return samples
    

    def sample_from_macrostate(self, n_sample, macrostate_id, ci_cutoff, weights='equilibrium'):
        ci = self.pcca_mod.memberships[:, macrostate_id]
        states_to_sample = ci > ci_cutoff

        if weights == 'equilibrium':
            stationary_distribution = self.msm_mod.stationary_distribution.copy()
            stationary_distribution[~states_to_sample] = 0
            weights = stationary_distribution / np.sum(stationary_distribution)
        elif weights == 'uniform':
            weights = np.zeros(self.msm_mod.n_states)
            weights[states_to_sample] = 1
            weights = weights / np.sum(weights)
        else:
            raise ValueError('weights should be either "stationary" or "uniform"')
        
        state_samples_count = self._get_state_count_from_distrib(weights, self.connected_states, n_sample)
        samples = self._get_samples_from_state(state_samples_count)
        
        return samples


    def sample_from_microstate(self, microstate_id, n_sample):
        weights = np.zeros(self.connected_states.shape[0])
        weights[microstate_id] = 1
        state_samples_count = self._get_state_count_from_distrib(weights, self.connected_states, n_sample)
        samples = self._get_samples_from_state(state_samples_count)
        
        return samples
        

    def _map_to_rtraj_samples(self, samples, dataset_keys):
        datasets_study = {key: self.traj_data.datasets[key] for key in dataset_keys} # Datasets used in this study 
        num_of_rtrajs_per_ds = {key:len(datasets_study[key]['rtraj_files']) for key in datasets_study.keys()}
        stride_of_rtraj_idx = np.concatenate([np.ones(num)*datasets_study[k]['stride'] for k, num in num_of_rtrajs_per_ds.items()])

        if isinstance(samples, list):
            rtraj_samples = []
            for ftraj_idx, frame_idx in samples:
                rtraj_idx = self.mapping[ftraj_idx]
                rframe_idx = frame_idx * stride_of_rtraj_idx[rtraj_idx]
                rtraj_samples.append([rtraj_idx, rframe_idx])
        elif isinstance(samples, dict):
            rtraj_samples = {}
            for ftraj_idx, frame_idx in samples.items():
                rtraj_idx = self.mapping[ftraj_idx]
                rframe_idx = [int(idx * stride_of_rtraj_idx[rtraj_idx]) for idx in frame_idx]
                rtraj_samples[rtraj_idx] = rframe_idx
        else:
            raise ValueError('samples should be either a list or a dictionary')
        return rtraj_samples


    def save_samples(self, samples, fname, ref=None, save_ids=True):
        dataset_keys = [f.strip() for f in self.hp_dict.datasets.lower().split(' ')] # Keys used in this study
        traj_files = np.concatenate([self.traj_data.datasets[key]['rtraj_files'] for key in dataset_keys])
    
        fname = Path(fname)

        traj_sample_dict = defaultdict(list)
        for traj_idx, frame_idx in samples:
            traj_sample_dict[traj_idx].append(frame_idx)
        traj_sample_dict = {int(k): [int(v) for v in vals] for k, vals in traj_sample_dict.items()}
        rtraj_sample_dict = self._map_to_rtraj_samples(traj_sample_dict, dataset_keys)
        
        if save_ids:
            with open(fname.with_suffix('.json'), 'w') as f:
                json.dump(traj_sample_dict, f)
            with open(fname.with_name(fname.stem + "_raw.json"), 'w') as f:
                json.dump(rtraj_sample_dict, f)
    
        frames = []
        for traj_id, frame_idx in rtraj_sample_dict.items():
            sample_traj = md.load(traj_files[traj_id])
            sample_frames = sample_traj[frame_idx].atom_slice(sample_traj.top.select('mass>1.1'))
            frames.append(sample_frames)
        if len(frames)>1:
            sampled_frames = md.join(frames)
        else:
            sampled_frames = frames[0]

        if ref is not None:
            try:
                sampled_frames = sampled_frames.superpose(ref, atom_indices=sampled_frames.top.select('name CA'))
            except:
                print('Wrong reference. Saving unsuperposed frames.')
        
        if isinstance(fname, Path):
            fname = fname.as_posix()
        sampled_frames.save(fname)
        print(f'Saved samples to {fname}')
    

    @property
    def hp_id(self):
        return self._hp_id


    @property
    def pcca_n(self):
        return self._pcca_n