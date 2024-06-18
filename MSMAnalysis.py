import os, sys
import warnings
from pathlib import Path
from typing import *
from natsort import natsorted
from tqdm import tqdm
from addict import Dict as Adict

import numpy as np
import pandas as pd

from MSMStudy import MSMStudy

warnings.simplefilter("always", UserWarning)


class MSMAnalysis():
    '''
    Parameters
    ----------
    protein : str
        The protein of interest. Must be either 'egfr' or 'abl'
    trajlen_cutoff : int
        The minimum length of the feature trajectories to be considered
    '''

    def __init__(self, protein, trajlen_cutoff=1000):
        if protein in ['egfr', 'abl']:
            self.protein = protein
        else:
            raise ValueError('Protein must be either "egfr" or "abl"')
        self._features = set()
        self._ftrajs_raw = dict()
        self.studies = dict()
        self.data = dict()
        self._trajlen_cutoff = trajlen_cutoff


    @property
    def features(self):
        '''
        Returns the feature trajectories that have been loaded
        '''
        return self._features


    @property
    def ftrajs_raw(self):
        return self._ftrajs_raw


    def __str__(self) -> str:
        return f'MSMAnalysis: {self.protein}\nFeatures: {self.features}\nStudies: {list(self.studies.keys())}'


    def load_ftrajs(self, features, arc_dir):
        '''
        Load the raw feature trajectories to build the MSMs from a given directory.

        Parameters
        ----------
        features : list
            A list of strings of features. Should correspond to the ftraj files in the directory
        arc_dir : str
            The directory where the ftrajs are stored
        '''

        if isinstance(arc_dir,str):
            arc_dir = Path(arc_dir)

        new_ftraj_files_ls = []
        new_feature_list = []
        for i, feature in enumerate(features):
            if feature in self._features and self._ftrajs_raw[feature] is not None:
                print(f'Feature {feature} already loaded. Skipping...')
                continue
            else:
                ftraj_files = natsorted([str(ftraj) for ftraj in arc_dir.rglob(f'run*-clone?_{feature}.npy')])
                if len(ftraj_files) == 0:
                    raise ValueError(f'No feature trajectories found for {feature}. Check the directory.')
                new_ftraj_files_ls.append(ftraj_files)
                new_feature_list.append(feature)

        ftraj_files_lens = [len(traj_files) for traj_files in new_ftraj_files_ls]
        if not all(len == ftraj_files_lens[0] for len in ftraj_files_lens):
            raise ValueError('Feature trajectories are not of equal length. Check if the feature trajectories exist.')

        for i, feature in enumerate(new_feature_list):
            print("Loading feature: ", feature)
            feature_ls = []
            for ftraj_file in tqdm(new_ftraj_files_ls[i], total=len(new_ftraj_files_ls[i])):
                new_ftraj = np.load(ftraj_file, allow_pickle=True)
                if new_ftraj.ndim == 1:
                    new_ftraj = new_ftraj[:, np.newaxis]
                feature_ls.append(new_ftraj)
            self._ftrajs_raw[feature] = feature_ls
        self._features.update(new_feature_list)
        print(f'Features loaded: {self._features}')


    def load_data(self, data_key, ftraj_dict):
        '''
        Load test data for the MSM analysis.

        Parameters
        ----------
        data_key : str
            The key to identify the test data
        ftraj_dict : dict
            A dictionary containing the feature trajectories for the test data
        '''

        if not hasattr(self, '_features'):
            raise ValueError('Please load the raw feature trajectories first.')
        
        for key, value in ftraj_dict.items():
            if key not in self._features:
                warnings.warn(f'Data feature {key} not found in the loaded feature trajectories to build MSM.', UserWarning)
            if data_key not in self.data.keys():
                self.data[data_key] = {}
            self.data[data_key][key] = value
    

    def select_ftrajs(self, features, data_key=None):
        '''
        Select the feature trajectories to be used for the MSM analysis.

        Parameters
        ----------
        features : list
            A list of strings of features to select the ftrajs 
        data_key : str
            The key to identify which dataset to select from. If None, the raw feature trajectories will be used
        
        Returns
        -------
        ftrajs : dict
            A dictionary containing the feature names and associated feature trajectories
        '''

        if data_key is None:
            if not hasattr(self, '_ftrajs_raw'):
                raise ValueError('Please load the raw feature trajectories first.')
            if not all(feature in self._ftrajs_raw.keys() for feature in features):
                raise ValueError('Some features not found in the loaded feature trajectories.')
            ftrajs = {feature:self._ftrajs_raw[feature] for feature in features}
        else:
            if not all(feature in self.data[data_key].keys() for feature in features):
                raise ValueError('Some features not found in the loaded data.')
            ftrajs = {feature:self.data[data_key][feature] for feature in features}

        return ftrajs 
    

    def create_study(self, study_name, hp_dict, features=None, stride=1, wk_dir=None, create_new=True,):
        '''
        Create a new MSM study with fixed hyperparameters and working directory. 

        Parameters
        ----------
        study_name : str
            Each study should contain a hyperparameter dictionary and a working directory
        hp_dict : dict
            A dictionary containing the hyperparameters for the MSM study
        features : list
            A list of strings of features to select the ftrajs. If None, all features will be used
        stride : int
            The stride to apply to the feature trajectories
        wk_dir : str
            The working directory for saving models and figures 
        '''

        if isinstance(wk_dir, str):
            wk_dir = Path(wk_dir)
        
        if wk_dir is None:
            wk_dir = Path.cwd() / 'data' / self.protein / 'msm'/ 'validation' / study_name

        if not wk_dir.is_dir():
            wk_dir.mkdir(parents=True, exist_ok=True)

        if features is None:
            features = self.features

        _selected_ftrajs = self.select_ftrajs(features)
        ftrajs, mapping = prepare_ftrajs(_selected_ftrajs, stride=stride, len_cutoff=self._trajlen_cutoff)
        self.studies[study_name] = MSMStudy(study_name, ftrajs, mapping, wk_dir, create_new=create_new, **hp_dict)

        return self.studies[study_name]


def prepare_ftrajs(ftrajs_dict, stride=1, len_cutoff=1000, convert_dihed=True):
    '''
    Concatenate and prepare the feature trajectories for the MSM analysis. 

    Parameters
    ----------
    ftrajs_dict : dict
        A dictionary containing the feature trajectories
    stride : int
        The stride to apply to the feature trajectories
    len_cutoff : int
        The minimum length of the feature trajectories to be considered

    Returns
    -------
    ftrajs : list
        A list of concatenated feature trajectories
    mapping : dict
        A dictionary mapping the indices of the selected feature trajectories to the original feature trajectories
    '''

    dihed_ids = np.where(['dihed' in k for k in list(ftrajs_dict.keys())])[0]
    ftrajs_raw = [v for v in ftrajs_dict.values()]
    no_of_trajs = len(ftrajs_raw[0])

    ftrajs, mapping = [], {}
    print("Preparing feature trajectories...")
    for i in tqdm(range(no_of_trajs), total=no_of_trajs):
        if ftrajs_raw[0][i].shape[0] < len_cutoff:
            continue

        ftrajs_to_add = [ftraj[i] for ftraj in ftrajs_raw]
        if convert_dihed:
            for id in dihed_ids:
                ftrajs_to_add[id] = np.concatenate([np.cos(ftrajs_to_add[id]), np.sin(ftrajs_to_add[id])], axis=1)

        ftrajs_to_add = np.concatenate(ftrajs_to_add, axis=1)
        ftrajs.append(ftrajs_to_add[::stride])
        mapping[len(ftrajs)-1] = i

    return ftrajs, mapping 
