from deeptime.markov.sample import compute_index_states
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


    @staticmethod
    def _load_numpy_series(directory):
        files = natsorted(Path(directory).glob('*.npy'))
        arrays = [np.load(f) for f in files]
        stacked = np.concatenate(arrays, axis=0) if arrays else np.empty((0,))
        return arrays, stacked


    @staticmethod
    def _read_pickle(path):
        with open(path, 'rb') as handle:
            return pickle.load(handle)


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

        dt_out = self.hp_dict['dt_out']
        for key, dataset in self.traj_data.datasets.items():
            stride = max(1, int(round(dt_out / dataset['dt'])))
            dataset['stride'] = stride
            print(f'Set dataset {key} stride to {stride}')

        print(f'Loading MSM model id {hp_id} from {self.save_dir}')
        print(f'{self.hp_dict}')

        self.load_all()


    def load_all(self):

        print('Loading trajectories...')
        self.ttrajs, self.ttraj_cat = self._load_numpy_series(self.save_dir / 'ttrajs')
        self.dtrajs, self.dtraj_cat = self._load_numpy_series(self.save_dir / 'dtrajs')

        print('Loading models...')
        model_dir = self.save_dir / 'models'
        with open(model_dir / 'mapping.json', 'rb') as f:
            self.mapping = {int(k): int(v) for k, v in json.load(f).items()}

        self.tica_mod = self._read_pickle(model_dir / 'tica_model.pkl')
        self.kmeans_mod = self._read_pickle(model_dir / 'kmeans_model.pkl')
        self.kmeans_centers = self.kmeans_mod.clustercenters
        self.count_mod = self._read_pickle(model_dir / 'count_model.pkl')

        if self.hp_dict.msm_mode == 'maximum_likelihood':
            self.msm_mod = self._read_pickle(model_dir / 'maximum_likelihood_msm_model.pkl')
        elif self.hp_dict.msm_mode == 'bayesian':
            self.baymsm_mod = self._read_pickle(model_dir / 'bayesian_msm_model.pkl')
            self.msm_mod = self.baymsm_mod.prior
        else:
            raise ValueError(f"Unknown msm_mode {self.hp_dict.msm_mode}")

        self.traj_weights = self.msm_mod.compute_trajectory_weights(self.dtrajs)
        self.connected_states = np.load(model_dir / 'connected_states.npy')
        self.disconnected_states = np.setdiff1d(np.arange(self.hp_dict.cluster_n), self.connected_states)
        self._index_states = compute_index_states(self.dtrajs)

        print('Done')


    def run_pcca(self, n):
        '''
        Run PCCA+ on the MSM model
        '''
        self._pcca_n = n
        self.pcca_mod = self.msm_mod.pcca(n)
        self.pcca_assignment = self.pcca_mod.assignments
        self.micro_to_macro = dict(zip(self.connected_states, self.pcca_assignment))
        mask = np.isin(self.dtraj_cat, self.connected_states)
        mapped = np.full_like(self.dtraj_cat, fill_value=-1)
        mapped[mask] = [self.micro_to_macro[idx] for idx in self.dtraj_cat[mask]]
        self.ptraj_cat = mapped


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
        dtraj = self.kmeans_mod.transform(ttraj)[0].ravel()

        connected_mask = np.isin(dtraj, self.connected_states)
        connected_d = dtraj[connected_mask]
        disconnected_d = dtraj[~connected_mask]
        pcca_assignment = np.array([self.micro_to_macro[d] for d in connected_d]) if connected_d.size else np.empty((0,), dtype=int)

        return ttraj, dtraj, connected_d, disconnected_d, pcca_assignment
    

    def _get_state_count_from_distrib(self, microstate_weights, connected_states, n_samples):
        """
        Decide how many sample to take from states according to microstate weights

        Parameters
        ----------
        microstate_weights: ndarray( (n) )
            A distribution over microstates to sample from
        n_samples: int
            The number of samples to be taken
        
        Returns
        -------
        state_samples_count: dict[int, int]
            The states to sampled from : the number of samples to be taken
        """

        assert len(microstate_weights) == len(connected_states), 'The distribution should have the same length as the connected states'

        sampled_states = np.random.choice(connected_states, size=n_samples, p=microstate_weights)
        unique_states, counts = np.unique(sampled_states, return_counts=True)

        return dict(zip(unique_states, counts))


    def _get_samples_from_state(self, state_samples_count):
        """
        Get samples from the states according to the number of samples to be taken
        Parameters
        ----------
        state_samples_count: dict[int, int]
            The states to sampled from : the number of samples to be taken
        Returns
        -------
        samples: list[tuple[int, int]]
            The samples to be taken. The samples are tuples of (ftraj_idx, frame_idx)
        """

        samples = []
        for state_to_sample_from, n_samples in state_samples_count.items():
            indices = np.array(self._index_states[state_to_sample_from])
            if n_samples > len(indices):
                raise ValueError(f'Requested {n_samples} samples from state {state_to_sample_from} which only has {len(indices)} frames')
            chooser = np.random.choice(len(indices), size=n_samples, replace=False)
            samples.append(indices[chooser])

        return np.concatenate(samples) if samples else np.empty((0, 2), dtype=int)
    

    def _get_all_frames_from_state(self, microstate_id):
        """
        Get all frames from a specific microstate
        
        Parameters
        ----------
        microstate_id: int
            The id of the microstate to get all frames from
            
        Returns
        -------
        samples: list[tuple[int, int]]
            All samples from the microstate. The samples are tuples of (ftraj_idx, frame_idx)
        """
        
        try:
            # Return all frames from this microstate
            samples = np.array(self._index_states[microstate_id])
        except KeyError:
            raise ValueError(f'Microstate {microstate_id} not found in the trajectory data')
        
        return samples


    def sample_from_distrib(self, n_sample, distrib):
        """
        Sample from the distribution using the connected states

        Parameters
        ----------
        n_sample: int
            The number of samples to be taken
        distrib: ndarray( (n) )
            A distribution over microstates to sample from

        Returns
        -------
        samples: list[tuple[int, int]]
            The samples to be taken. The samples are tuples of (ftraj_idx, frame_idx)
        """

        state_samples_count = self._get_state_count_from_distrib(distrib, self.connected_states, n_sample)
        samples = self._get_samples_from_state(state_samples_count)

        return samples
    

    def sample_from_macrostate(self, n_sample, macrostate_id, ci_cutoff, weights='equilibrium'):
        """
        Sample from a macrostate using the PCCA+ assignments

        Parameters
        ----------
        n_sample: int
            The number of samples to be taken
        macrostate_id: int
            The id of the macrostate to be sampled from
        ci_cutoff: float
            The cutoff for the PCCA+ membership probabilities to be considered in the sampling
        weights: str
            The weights to be used for the sampling. Can be 'equilibrium' or 'uniform'

        Returns
        -------
        samples: list[tuple[int, int]]
            The samples to be taken. The samples are tuples of (ftraj_idx, frame_idx)
        """

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
            raise ValueError('weights should be either "equilibrium" or "uniform"')
        
        state_samples_count = self._get_state_count_from_distrib(weights, self.connected_states, n_sample)
        samples = self._get_samples_from_state(state_samples_count)
        
        return samples


    def sample_from_microstate(self, microstate_id, n_sample=None):
        """
        Sample from a microstate using the PCCA+ assignments

        Parameters
        ----------
        microstate_id: int
            The id of the microstate to be sampled from
        n_sample: int, optional
            The number of samples to be taken. If None, returns all frames in the microstate.
            
        Returns
        -------
        samples: list[tuple[int, int]]
            The samples to be taken. The samples are tuples of (ftraj_idx, frame_idx)
        """

        if n_sample is None:
            # Return all frames from this microstate
            samples = self._get_all_frames_from_state(microstate_id)
        else:
            # Use existing sampling logic
            weights = np.zeros(self.connected_states.shape[0])
            microstate_idx = np.where(self.connected_states == microstate_id)[0]
            if len(microstate_idx) == 0:
                raise ValueError(f'Microstate {microstate_id} not found in connected states')
            weights[microstate_idx[0]] = 1
            state_samples_count = self._get_state_count_from_distrib(weights, self.connected_states, n_sample)
            samples = self._get_samples_from_state(state_samples_count)
        
        return samples
        

    def _map_to_rtraj_samples(self, samples, dataset_keys):
        """
        Map the sample indices from the filtered featurized trajectories to the indices of raw trajectories

        Parameters
        ----------
        samples: list[tuple[int, int]] or dict[int, list[int]]
            The samples to be mapped. The samples are tuples of (ftraj_idx, frame_idx) or dict[ftraj_idx, list[frame_idx]]
        dataset_keys: list[str]
            The keys of the datasets used in this study
        Returns
        -------
        rtraj_samples: list[tuple[int, int]] or dict[int, list[int]]
            The mapped samples. The samples are tuples of (rtraj_idx, rframe_idx)
        """

        datasets_study = [self.traj_data.datasets[key] for key in dataset_keys]
        strides = []
        for dataset in datasets_study:
            stride = dataset.get('stride', 1)
            strides.extend([stride] * len(dataset['rtraj_files']))
        stride_lookup = np.asarray(strides, dtype=int) if strides else np.array([], dtype=int)

        if isinstance(samples, list):
            sample_dict = defaultdict(list)
            for ftraj_idx, frame_idx in samples:
                sample_dict[ftraj_idx].append(frame_idx)
        elif isinstance(samples, dict):
            sample_dict = samples
        else:
            raise ValueError('samples should be either a list or a dictionary')

        rtraj_samples = {}
        for ftraj_idx, frame_idx in sample_dict.items():
            rtraj_idx = self.mapping[ftraj_idx]
            stride = stride_lookup[rtraj_idx] if stride_lookup.size else 1
            scaled = [int(idx * stride) for idx in np.atleast_1d(frame_idx)]
            rtraj_samples.setdefault(rtraj_idx, []).extend(scaled)

        if isinstance(samples, list):
            return [[traj_idx, idx] for traj_idx, indices in rtraj_samples.items() for idx in indices]

        return {traj_idx: sorted(set(indices)) for traj_idx, indices in rtraj_samples.items()}


    def save_samples(self, samples, fname, ref=None, save_ids=True, concat=True):
        """
        Extract the sampled frames from the raw trajectories using sample indices and save them to a file with MDAnalysis

        Parameters
        ----------
        samples: list[tuple[int, int]]
            The samples to be saved. The samples are tuples of (ftraj_idx, frame_idx)
        fname: str or Path
            The name of the file to save the samples to (if concat=True) or directory path (if concat=False)
        ref: mdtraj.Trajectory, optional
            The reference trajectory to superpose the sampled frames to. The default is None.
        save_ids: bool, optional
            Whether to save the sampled indices to a json file. The default is True.
        concat: bool, optional
            Whether to concatenate all samples into one file (True) or save as separate numbered files (False). 
            If False, fname should be a directory path. The default is True.
        """

        dataset_keys = [f.strip() for f in self.hp_dict.datasets.lower().split(' ')] # Keys used in this study
        traj_files = np.concatenate([self.traj_data.datasets[key]['rtraj_files'] for key in dataset_keys])
    
        fname = Path(fname)

        traj_sample_dict = defaultdict(list)
        for traj_idx, frame_idx in samples:
            traj_sample_dict[traj_idx].append(frame_idx)
        traj_sample_dict = {int(k): [int(v) for v in vals] for k, vals in traj_sample_dict.items()}
        rtraj_sample_dict = self._map_to_rtraj_samples(traj_sample_dict, dataset_keys)
        
        if save_ids:
            if concat:
                with open(fname.with_suffix('.json'), 'w') as f:
                    json.dump(traj_sample_dict, f)
                with open(fname.with_name(fname.stem + "_raw.json"), 'w') as f:
                    json.dump(rtraj_sample_dict, f)
            else:
                # Create directory if it doesn't exist
                fname.mkdir(parents=True, exist_ok=True)
                with open(fname / 'all_samples.json', 'w') as f:
                    json.dump(traj_sample_dict, f)
                with open(fname / 'all_samples_raw.json', 'w') as f:
                    json.dump(rtraj_sample_dict, f)

        if concat:
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
                fname_str = fname.as_posix()
            else:
                fname_str = fname
            sampled_frames.save(fname_str)
            print(f'Saved concatenated samples to {fname_str}')
        
        else:
            # Save each frame as a separate file
            frame_count = 0
            fname.mkdir(parents=True, exist_ok=True)
            for traj_id, frame_idx in rtraj_sample_dict.items():
                print(f'Sampling trajectory {traj_files[traj_id]}')
                sample_traj = md.load(traj_files[traj_id])
                sample_frames = sample_traj[frame_idx].atom_slice(sample_traj.top.select('mass>1.1'))
                if ref is not None:
                    try:
                        sample_frames = sample_frames.superpose(ref, atom_indices=sample_frames.top.select('name CA'))
                    except:
                        print('Wrong reference. Saving unsuperposed frame.')
                sample_file = fname / f'frames_{frame_count:04d}.pdb'
                sample_frames.save(sample_file.as_posix())
                frame_count += 1
                    
    
    @property
    def hp_id(self):
        return self._hp_id


    @property
    def pcca_n(self):
        return self._pcca_n