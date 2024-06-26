from pyemma.coordinates import tica 
from pyemma.coordinates import cluster_kmeans
import pyemma as pm

from deeptime.markov import TransitionCountEstimator
from deeptime.markov.msm import MaximumLikelihoodMSM
from deeptime.markov.msm import BayesianMSM
from deeptime.markov.tools import *
import deeptime as dt

import pickle
import numpy as np
from pathlib import Path


class MSMStudy():
    _allowed_paramters = ['tica__lag', 'tica__stride', 'tica__dim',             # TICA parameters
                          'cluster__k', 'cluster__stride', 'cluster__maxiter',  # Clustering parameters
                          'markov__lag', 'count__mode',                         # Transition count parameters    
                          'markov__mode', 'pcca_n'
                          ]

    def __init__(self, study_name, ftrajs, mapping, wk_dir, create_new=True, **kwargs):
        self._seed = 42
        self.study_name = study_name
        self.ftrajs = ftrajs
        self.mapping = mapping
        self.wk_dir = wk_dir
        self.model_dir = self.wk_dir / 'models'
        self.fig_dir = self.wk_dir / 'plots'
        self.sample_dir = self.wk_dir / 'samples'        

        if create_new:
            for key, value in kwargs.items():
                if key in self._allowed_paramters:
                    internal_key = f"_{key}"
                    setattr(self, internal_key, value)
                else:
                    print(f'Parameter {key} not allowed, ignore. Check the allowed parameters')
                    continue
            
            if '_tica__lag' not in self.__dict__:
                self._tica__lag = 1
                print('TICA lag time not specified. Defaulting to 1')
            
            if '_tica__stride' not in self.__dict__:
                self._tica__stride = 1
                print('Tica stride not specified. Defaulting to 1')
            
            if '_tica__dim' not in self.__dict__:
                self._tica__dim = 10
                print('Tica dimension not specified. Defaulting to 2')
            
            if '_cluster__k' not in self.__dict__:
                self._cluster__k = 100
                print('Number of clusters not specified. Defaulting to 100')

            if '_cluster__stride' not in self.__dict__:
                self._cluster__stride = 1
                print('Cluster stride not specified. Defaulting to 1')

            if '_cluster__maxiter' not in self.__dict__: 
                self._cluster__maxiter = 100
                print('Cluster max iterations not specified. Defaulting to 100')

            if '_markov__lag' not in self.__dict__:
                self._markov__lag = 1
                print('Markov lag time not specified. Defaulting to 1')
            
            if '_count__mode' not in self.__dict__:
                self._count__mode = 'effective'
                print('Count mode not specified. Defaulting to effective')
            
            if '_markov__mode' not in self.__dict__:
                self._markov__mode = 'bayesian'
                print('Markov mode not specified. Defaulting to bayesian')

            if '_pcca_n' not in self.__dict__:
                self._pcca_n = 5
                print('Number of metastable states not specified. Defaulting to 5. Run set_pcca() to set the number of metastable states')
            
            if not self.fig_dir.exists():
                self.fig_dir.mkdir(parents=True)
                
            if not self.sample_dir.exists():
                self.sample_dir.mkdir(parents=True)

            #self._save_model_params()
        else:
            self.restore_models()

        self.__str__()
        

    def __str__(self) -> str:
        return f'MSMStudy: {self.study_name}\nWorking directory: {self.wk_dir}\nParameters: {self.get_paramters()}'


    @property
    def pcca_n(self):
        return self._pcca_n
    

    def get_paramters(self):
        return ', '.join(f"{k}={v}" for k, v in self.__dict__.items() if k.strip('_') in self._allowed_paramters)
    

    def set_pcca(self, n):
        if not isinstance(n, int):
            raise ValueError('Number of metastable states must be an integer')
        if hasattr(self, '_pcca_n') and hasattr(self, 'pcca_mod'):
            print('Overwriting previous PCCA model')
            self._pcca_n = n
            self._pcca()
        else:
            self._pcca_n = n
            print('Set number of metastable states to ', n)
        self._save_model_params()


    def _tica(self):
        self.tica_mod = tica(self.ftrajs, lag=self._tica__lag, stride=self._tica__stride, 
                             dim=self._tica__dim, kinetic_map=True)
        self.ttrajs = self.tica_mod.get_output()
        self.ttraj_cat = np.concatenate(self.ttrajs)


    def _kmeans(self):
        if not self.ttrajs:
            raise ValueError('TICA trajectories not found. Run TICA first')
        self.kmeans_mod = cluster_kmeans(data=None, k=self._cluster__k, max_iter=self._cluster__maxiter, 
                                         stride=self._cluster__stride, fixed_seed=self._seed)
        self.dtrajs = [dtraj.flatten() for dtraj in self.kmeans_mod.fit_transform(self.ttrajs)]
        self.dtraj_cat = np.concatenate(self.dtrajs, axis=0)
        self.kmeans_centers = self.kmeans_mod.clustercenters


    def _trans_count(self):
        if not self.dtrajs:
            raise ValueError('Discrete trajectories not found. Run kmeans clustering first')
        self.count_mod = TransitionCountEstimator(lagtime=self._markov__lag, count_mode=self._count__mode).fit_fetch(self.dtrajs)
        self.connected_states = estimation.largest_connected_set(self.count_mod.count_matrix)
        self.disconnected_states = np.setdiff1d(np.arange(self._cluster__k), self.connected_states)
        if self.disconnected_states.shape[0] > 0:
            print(f'Found disconnected states: {self.disconnected_states}')


    def _ML_MSM(self):
        if not self.count_mod:
            raise ValueError('Transition count not found. Run transition count estimation first')
        self.msm_mod = MaximumLikelihoodMSM(reversible=True).fit_fetch(self.count_mod)
        self.traj_weights = self.msm_mod.compute_trajectory_weights(self.dtrajs)


    def _bayesian_MSM(self):
        if not self.count_mod:
            raise ValueError('Transition count not found. Run transition count estimation first')
        self.baymsm_mod = BayesianMSM(reversible=True).fit_fetch(self.count_mod)
        self.msm_mod = self.baymsm_mod.prior
        self.traj_weights = self.msm_mod.compute_trajectory_weights(self.dtrajs)


    def _pcca(self):
        if not self.msm_mod:
            raise ValueError('MSM not found. Run MSM estimation first')
        if not hasattr(self, '_pcca_n'):
            raise ValueError('Number of metastable states not set. Run set_pcca() to set the number of metastable states')
        self.pcca_mod = self.msm_mod.pcca(n_metastable_sets=self._pcca_n)
        self.pcca_assignment = self.pcca_mod.assignments
        self.micro_to_macro = {self.connected_states[i]: self.pcca_assignment[i] for i in range(len(self.connected_states))}
        self.ptraj_cat = np.array([self.micro_to_macro[d] if d in self.connected_states else -1 for d in self.dtraj_cat])


    def estimate_MSM(self):
        self._tica()
        self._kmeans()
        self._trans_count()
        if self._markov__mode == 'bayesian':
            self._bayesian_MSM()
        elif self._markov__mode == 'ml':
            self._ML_MSM()
        else:
            raise ValueError('Markov mode not recognised. Choose between "bayesian" or "ml"')
        self._pcca()
        print('MSM estimation complete')


    def transform(self, data_ftraj):
        if not self.tica_mod:
            raise ValueError('TICA model not found. Run estimate_MSM or restore_models first.')
        if not self.kmeans_mod:
            raise ValueError('Kmeans model not found. Run estimate_MSM or restore_models first.')
        if not self.msm_mod:
            raise ValueError('MSM model not found. Run estimate_MSM or restore_models first.')
        if not self.pcca_mod:
            raise ValueError('PCCA model not found. Run estimate_MSM or restore_models first.')
        
        ttraj = self.tica_mod.transform(data_ftraj)
        dtraj = self.kmeans_mod.transform(ttraj)[0].flatten()
        connected_d = np.array([idx for idx in dtraj if idx in self.connected_states]).flatten()
        disconnected_d = np.array([idx for idx in dtraj if idx in self.disconnected_states]).flatten()
        pcca_assignment = np.array([self.micro_to_macro[d] for d in connected_d])

        return ttraj, dtraj, connected_d, disconnected_d, pcca_assignment


    def save_models(self, overwrite=True):
        self._save_model_params()

        self.tica_mod.save(str(self.model_dir / 'tica_model'), overwrite=overwrite)
        self.kmeans_mod.save(str(self.model_dir / 'kmeans_model'), overwrite=overwrite)
        with open(self.model_dir/'count_model.pkl', 'wb') as file:
            pickle.dump(self.count_mod, file)
        if hasattr(self, 'baymsm_mod'):
            with open(self.model_dir/'bayesian_msm.pkl', 'wb') as file:
                pickle.dump(self.baymsm_mod, file)
        elif hasattr(self, 'msm_mod'):
            if overwrite:
                np.save(self.model_dir/'transition_matrix.npy', self.msm_mod.transition_matrix)
            else:
                raise ValueError('Overwriting transition matrix not allowed. Set overwrite=True to overwrite')
    

    def _save_model_params(self):
        if not self.model_dir.exists():
            self.model_dir.mkdir(parents=True)

        with open(self.model_dir / 'study_params.txt', 'w') as f:
            f.write(f'{self.study_name}\n')
            for key, val in self.__dict__.items():
                if key.strip('_') in self._allowed_paramters:
                    f.write(f'{key}:{val}\n')
        

    def restore_models(self):
        if not self.model_dir.exists():
            raise ValueError('Model directory not found. Check the working directory')
        
        with open(self.model_dir / 'study_params.txt', 'r') as f:
            self.study_name = f.readline().strip('\n')
            print(f'Restoring study: {self.study_name}')
            for line in f:
                key, val = line.split(':')
                if key.strip('_') in self._allowed_paramters:
                    try:
                        self.__dict__[key] = int(val.strip('\n'))
                    except:
                        self.__dict__[key] = val.strip('\n')
        
        if (self.model_dir / 'tica_model').exists():
            print('Restoring TICA model')
            tica_mod = pm.coordinates.tica()
            self.tica_mod = tica_mod.load(self.model_dir/'tica_model')
            self.ttrajs = self.tica_mod.transform(self.ftrajs)
            self.ttraj_cat = np.concatenate(self.ttrajs, axis=0)
        else:
            raise ValueError('TICA model not found. Check the working directory')
        
        if (self.model_dir / 'kmeans_model').exists():
            print('Restoring Kmeans model')
            kmeans_mod = pm.coordinates.cluster_kmeans()
            self.kmeans_mod = kmeans_mod.load(self.model_dir/'kmeans_model')
            self.dtrajs = [dtraj.flatten() for dtraj in self.kmeans_mod.transform(self.ttrajs)]
            self.dtraj_cat = np.concatenate(self.dtrajs, axis=0)
            self.kmeans_centers = self.kmeans_mod.clustercenters
        else:
            raise ValueError('Kmeans model not found. Check the working directory')
        
        if (self.model_dir / 'count_model.pkl').exists():
            print('Restoring transition count model')
            with open(self.model_dir / 'count_model.pkl', 'rb') as file:
                self.count_model = pickle.load(file)
            self.connected_states = estimation.largest_connected_set(self.count_model.count_matrix)
            self.disconnected_states = np.setdiff1d(np.arange(self._cluster__k), self.connected_states)
            if self.disconnected_states.shape[0] > 0:
                print(f'Found disconnected states: {self.disconnected_states}')
        else:
            raise ValueError('Transition count model not found. Check the working directory')

        if self._markov__mode == 'bayesian':
            if (self.model_dir / 'bayesian_msm.pkl').exists():
                print('Restoring Bayesian MSM model')
                with open(self.model_dir / 'bayesian_msm.pkl', 'rb') as file:
                    self.baymsm_mod = pickle.load(file)
                    self.msm_mod = self.baymsm_mod.prior
                    self.traj_weights = self.msm_mod.compute_trajectory_weights(self.dtrajs)
            else:
                raise ValueError('Bayesian MSM model not found. Check the working directory')
        elif self._markov__mode == 'ml':
            self._ML_MSM()

        self._pcca()
        print('Models restore complete.')