from pathlib import Path
from time import time
import json
import os

import numpy as np
import pandas as pd
import pickle
import pyemma as pm
from addict import Dict as Adict
from tqdm import tqdm

from funcs_count import PriorTransitionCountEstimator
from deeptime.markov.msm import BayesianMSM, MaximumLikelihoodMSM
from deeptime.markov.tools import estimation


class MSMEstimation():
    '''
    This class estimates MSMs using TrajData and hyperparameter dictionaries.
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
            The directory to save the MSMs and the estimated hyperparameters
        '''

        self.wk_dir = Path(wk_dir)
        self.wk_dir.mkdir(parents=True, exist_ok=True)
        self.observation_path = self.wk_dir / 'observation.csv'
        self.hps_table = hps_table
        self.traj_data = traj_data


    def _prepare_save_dirs(self, hp_id):
        save_dir = self.wk_dir / f'{hp_id}'
        model_dir = save_dir / 'models'
        ttraj_dir = save_dir / 'ttrajs'
        dtraj_dir = save_dir / 'dtrajs'

        for directory in (save_dir, model_dir, ttraj_dir, dtraj_dir):
            directory.mkdir(parents=True, exist_ok=True)

        return save_dir, model_dir, ttraj_dir, dtraj_dir


    @staticmethod
    def _write_params(save_dir, hp_dict):
        with open(save_dir / 'params.json', 'w') as f:
            json.dump(dict(hp_dict), f, indent=4)


    def _study_exists(self, hp_dict, model_dir):
        if not self.observation_path.exists():
            return False

        try:
            observation = pd.read_csv(self.observation_path, usecols=['hp_id'])
        except ValueError:
            observation = pd.read_csv(self.observation_path)

        if 'hp_id' not in observation.columns:
            return False

        if hp_dict.hp_id not in observation['hp_id'].values:
            return False

        required = [
            model_dir / 'mapping.json',
            model_dir / 'tica_model.pkl',
            model_dir / 'kmeans_model.pkl',
            model_dir / 'count_model.pkl',
        ]
        msm_file = model_dir / ('maximum_likelihood_msm_model.pkl' if hp_dict.msm_mode == 'maximum_likelihood' else 'bayesian_msm_model.pkl')
        required.append(msm_file)

        return all(path.exists() for path in required)


    def run_studies(self, hp_indices):
        '''
        Iterate over the hyperparameter table and run MSM estimation for each hyperparameter dict
        Skip the already existing studies by checking if the model files exist

        Parameters
        ----------
        hp_indices : list
            List of hyperparameter indices to run
        '''

        hptable_to_run = self.hps_table.loc[self.hps_table['hp_id'].isin(hp_indices)]
        print('No of hp trials: ', len(hp_indices))
        for _, row in tqdm(hptable_to_run.iterrows(), total=len(hp_indices)):
            hp_dict = Adict(row.to_dict())
            save_dir, model_dir, ttraj_dir, dtraj_dir = self._prepare_save_dirs(hp_dict.hp_id)

            if self._study_exists(hp_dict, model_dir):
                print(f'Study {hp_dict.hp_id} already exists. Skipping ...')
                continue

            print('Processing trial:\n', hp_dict)
            self._write_params(save_dir, hp_dict)
            ftrajs, _ = self.get_ftrajs(hp_dict, model_dir)
            self.estimate_msm(ftrajs, hp_dict, self.wk_dir, model_dir, ttraj_dir, dtraj_dir)


    def get_ftrajs(self, hp_dict, model_dir):
        '''
        Get the feature trajectories for the hyperparameter dictionary

        Parameters
        ----------
        hp_dict : Adict
            Hyperparameter dictionary
        
        Returns
        -------
        ftrajs : List[np.ndarray]
            List of feature trajectories
        mapping : dict
            Mapping from the filtered traj indices to the original indices
        '''

        datasets = [f.strip() for f in hp_dict.datasets.lower().split(' ')]
        features = [f.strip() for f in hp_dict.features.lower().split(' ')]
        dihed_ids = np.where([('dihed' in f) and ('dihedgroup' not in f) for f in features])[0]
        
        for dataset in datasets:
            self.traj_data.load_ftrajs(key=dataset,
                                       feature_names=features)
        ftrajs, mapping = self.traj_data.get_ftrajs(keys=datasets,
                                                    dt_out=hp_dict.dt_out,
                                                    internal_names=features, 
                                                    time_cutoff=hp_dict.time_cutoff,
                                                    convert_dihed_ids=dihed_ids)
        with open(model_dir/'mapping.json', 'w') as f:
            json.dump(mapping, f, indent=4)
        return ftrajs, mapping


    @staticmethod
    def estimate_msm(ftrajs, hp_dict, obs_dir, model_dir, ttraj_dir, dtraj_dir):
        # Create observation output dictionaries that contains:
        # index of the study; whether the count matrix is sparase; eigenvalues; vamp2 scores
        n_score = 10
        imputation = np.float64(-1)

        heading = dict()
        heading['hp_id'] = hp_dict.hp_id
        heading['time_consumed'] = imputation

        sparse_dict = {'is_sparse': False}
        ev_dict = {f'ev_{j+1}': imputation for j in range(0, n_score)}
        ev_std_dict = {f'ev_std_{j+1}': imputation for j in range(0, n_score)}
        ts_dict = {f't_{j+2}': imputation for j in range(0, n_score)}
        ts_std_dict = {f't_std_{j+2}': imputation for j in range(0, n_score)}
        vamp2_dict = {f'vamp2_{j+2}': imputation for j in range(0, n_score)}
        vamp2_std_dict = {f'vamp2_std_{j+2}': imputation for j in range(0, n_score)}

        # Estimate the MSM. If the estimation fails, skip this study and save the heading.
        try:
            t_start = time()
            ttrajs, tica_mod = MSMEstimation.tica(hp_dict, ftrajs, model_dir, ttraj_dir)
            dtrajs, kmeans_mod = MSMEstimation.kmeans(hp_dict, ttrajs, model_dir, dtraj_dir)
            count_mod = MSMEstimation.count(hp_dict, dtrajs, model_dir)
            msm_mod = MSMEstimation.msm(hp_dict, count_mod, model_dir)
            t_elapsed = time() - t_start
        except Exception as e:
            print('MSM Estimation has failed. Save the heading ...')
            with open("error_log.txt", "a") as file:
                file.write(f"{hp_dict}\nError occurred: {e}\n")            
            results = {**heading, **sparse_dict, **ev_dict, **ev_std_dict, **ts_dict, **ts_std_dict,
                       **vamp2_dict, **vamp2_std_dict}   
            data = pd.DataFrame([results])
            data.to_csv(obs_dir/f'observation.csv', index=False, mode='a', header=not os.path.exists(obs_dir/f'observation.csv'))
            return None 
        
        # Assemble the observation dicts
        heading['time_consumed'] = t_elapsed
        # If the model is Bayesian, report the results of the prior (maximum likelihood) model and collect the standard deviations
        if hp_dict.msm_mode == 'bayesian':
            no = min(msm_mod.prior.transition_matrix.shape[0]-1, n_score)
            sparse_dict = {'is_sparse' : msm_mod.prior.transition_matrix.shape[0] != hp_dict.cluster_n}
            for k in range(no):
                ev_dict[f'ev_{k+1}'] = msm_mod.prior.eigenvalues()[k]
                ev_std_dict[f'ev_std_{k+1}'] = msm_mod.gather_stats('eigenvalues').std[k]
                ts_dict[f't_{k+2}'] = msm_mod.prior.timescales()[k]
                ts_std_dict[f't_{k+2}'] = msm_mod.gather_stats('timescales').std[k]
                vamp2_dict[f'vamp2_{k+2}'] = msm_mod.prior.score(dtrajs, r=2, dim=k+2)
                vamp2_std_dict[f'vamp2_std_{k+2}'] = msm_mod.gather_stats('score', dtrajs=dtrajs, r=2, dim=k+2).std
        # If the model is maximum likelihood, report the results of the maximum likelihood model
        elif hp_dict.msm_mode == 'maximum_likelihood':
            no = min(msm_mod.transition_matrix.shape[0]-1, n_score)
            sparse_dict = {'is_sparse' : msm_mod.transition_matrix.shape[0] != hp_dict.cluster_k}
            for k in range(no):
                ev_dict[f'ev_{k+1}'] = msm_mod.eigenvalues()[k]
                ts_dict[f't_{k+2}'] = msm_mod.timescales()[k]
                vamp2_dict[f'vamp2_{k+2}'] = msm_mod.score(dtrajs, r=2, dim=k+2)

        # Save the observations to csv
        print('Saving results ...')
        results = {**heading, **sparse_dict, **ev_dict, **ev_std_dict, **ts_dict, **ts_std_dict, **vamp2_dict, **vamp2_std_dict}   
        data = pd.DataFrame([results])
        data.to_csv(obs_dir/f'observation.csv', index=False, mode='a', header= not os.path.exists(obs_dir/f'observation.csv'))
        return None

    
    @staticmethod
    def tica(hp_dict, ftrajs, model_dir=None, ttraj_dir=None):
        '''
        Perform TICA on the feature trajectories.

        Parameters
        ----------
        hp_dict : Adict
            Hyperparameter dictionary.
        ftrajs : List[np.ndarray]
            List of feature trajectories.

        Returns
        -------
        ttrajs : List[np.ndarray]
            List of TICA-transformed trajectories.
        tica_mod : pm.coordinates.tica
            TICA model.
        '''

        lag = hp_dict.tica_lag_time / hp_dict.dt_out
        if not lag.is_integer():
            print(f'Warning: tICA lag time {lag} is not an integer. Rounding to the nearest integer {round(lag)}...')
            lag = int(round(lag))
        else:
            lag = int(lag)
        stride = hp_dict.tica_stride
        dim = hp_dict.tica_dim
        tica_kinetic_map = hp_dict.tica_kinetic_map

        tica_mod = pm.coordinates.tica(ftrajs, lag=lag, stride=stride, dim=dim, kinetic_map=tica_kinetic_map)
        ttrajs = tica_mod.get_output()

        if model_dir is not None:
            with open(model_dir/'tica_model.pkl', 'wb') as f:
                pickle.dump(tica_mod, f)
        if ttraj_dir is not None:
            for i, ttraj in enumerate(ttrajs):
                np.save(ttraj_dir/f'ttraj_{i}.npy', ttraj)

        return ttrajs, tica_mod


    @staticmethod
    def kmeans(hp_dict, ttrajs, model_dir=None, dtraj_dir=None):
        """
        Perform K-means clustering on the TICA-transformed trajectories.

        Parameters
        ----------
        hp_dict : Adict
            Hyperparameter dictionary.
        ttrajs : List[np.ndarray]
            List of TICA-transformed trajectories.
        seed : int
            Random seed.
        
        Returns
        -------
        dtrajs : List[np.ndarray]
            List of discrete trajectories.
        kmeans_mod : pm.coordinates.cluster_kmeans
            K-means model.
        """

        n_clusters = hp_dict.cluster_n
        stride = hp_dict.cluster_stride
        max_iter = hp_dict.cluster_max_iter
        seed = hp_dict.seed

        kmeans_mod = pm.coordinates.cluster_kmeans(ttrajs, k=n_clusters, max_iter=max_iter, stride=stride, fixed_seed=seed)
        dtrajs = kmeans_mod.dtrajs

        if model_dir is not None:
            with open(model_dir/'kmeans_model.pkl', 'wb') as f:
                pickle.dump(kmeans_mod, f)
        if dtraj_dir is not None:
            for i, dtraj in enumerate(dtrajs):
                np.save(dtraj_dir/f'dtraj_{i}.npy', dtraj)

        return dtrajs, kmeans_mod


    @staticmethod
    def count(hp_dict, dtrajs, model_dir=None):
        """
        Estimate the transition count matrix.

        Parameters
        ----------
        hp_dict : Adict
            Hyperparameter dictionary.
        dtrajs : List[np.ndarray]
            List of discrete trajectories.
        
        Returns
        -------
        count_mod : deeptime.markov.TransitionCountEstimator
            Transition count estimator.
        """

        lag = hp_dict.markov_lag_time / hp_dict.dt_out
        if not lag.is_integer():
            print(f'Warning: Markov lag time {lag} is not an integer. Rounding to the nearest integer {round(lag)}...')
            lag = int(round(lag))
        else:
            lag = int(lag)

        count_mode = hp_dict.markov_count_mode
        prior_count = hp_dict.markov_count_prior
        count_mod = PriorTransitionCountEstimator(lagtime=lag, count_mode=count_mode, prior=prior_count).fit_fetch(dtrajs)
        np.save(model_dir/'connected_states.npy', estimation.largest_connected_set(count_mod.count_matrix))

        if model_dir is not None:
            with open(model_dir/'count_model.pkl', 'wb') as f:
                pickle.dump(count_mod, f)

        return count_mod
    

    @staticmethod
    def msm(hp_dict, count_mod, model_dir=None):
        """
        Estimate the MSM.

        Parameters
        ----------
        hp_dict : Adict
            Hyperparameter dictionary.
        model_dir : Path
            Directory to save the model.
        
        Returns
        -------
        msm_mod : deeptime.markov.msm.MaximumLikelihoodMSM
            MSM model.
        """

        if hp_dict.msm_mode == 'bayesian':
            msm_mod = BayesianMSM(reversible=True).fit_fetch(count_mod)
            with open(model_dir/'bayesian_msm_model.pkl', 'wb') as f:
                pickle.dump(msm_mod, f)
        elif hp_dict.msm_mode == 'maximum_likelihood':
            msm_mod = MaximumLikelihoodMSM(reversible=True).fit_fetch(count_mod)
            with open(model_dir/'maximum_likelihood_msm_model.pkl', 'wb') as f:
                pickle.dump(msm_mod, f)

        return msm_mod

