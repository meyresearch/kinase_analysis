from pyemma.coordinates import tica 
from pyemma.coordinates import cluster_kmeans
import pyemma as pm

from deeptime.markov import TransitionCountEstimator
from deeptime.markov.msm import MaximumLikelihoodMSM
from deeptime.markov.msm import BayesianMSM
from deeptime.markov.tools import estimation
import deeptime as dt

import json
import pickle
import numpy as np
import pandas as pd
from addict import Dict as Adict
from pathlib import Path
from natsort import natsorted


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
    

    @property
    def hp_id(self):
        return self._hp_id


    @property
    def pcca_n(self):
        return self._pcca_n