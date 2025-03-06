from pathlib import Path
from natsort import natsorted
from tqdm import tqdm
import numpy as np
import re 
import os
import json
import mdtraj as md


class TrajData():
    '''
    This class is an I/O for raw and featurised trajectories for MSM estimation and analysis.
    '''

    def __init__(self, protein):
        '''
        Parameters
        ----------
        protein : str
            Name of the protein simulations
        '''

        self.protein = protein
        self.datasets = dict()
        self._ftrajs = dict()
    

    def add_dataset(self, key, rtraj_dir, ftraj_dir, dt, rtraj_glob='*'):
        '''
        Set a simulation dataset by setting the raw and featurised trajectory directories.
        
        Parameters
        ----------
        key : str
            Key to access the simulation dataset
        rtraj_dir : str or Path
            Directory of the raw simulation trajectories
        ftraj_dir : str or Path
            Directory of the featurised simulation trajectories
        dt : float
            Time step of the simulation trajectories in ns
        glob : str or None
            Glob pattern to search for the raw trajectory files. By default all files in the directory are considered as trajectories.
        '''

        if key in self.datasets:
            print(f'Key <{key}> already exists in the dataset.')
            return None
        
        rtraj_dir = Path(rtraj_dir)
        ftraj_dir = Path(ftraj_dir)
        rtraj_files = natsorted([traj for traj in rtraj_dir.glob(rtraj_glob)])

        if len(rtraj_files) == 0:
            raise ValueError(f'No raw trajectory files found in {rtraj_dir}. Check the directory or glob pattern.')
        
        print(f'Setting dataset <{key}>. \nNumber of raw trajectories: {len(rtraj_files)}\n')
        self.datasets[key] = {'rtraj_dir': rtraj_dir,
                             'ftraj_dir': ftraj_dir,
                             'rtraj_files': rtraj_files,
                             'dt': dt}
        self._ftrajs[key] = dict()


    def __str__(self):
        return f'TrajData for: {self._protein}\nDatasets: {list(self.datasets.keys())}'


    def featurize(self, key, featurisers, feature_names=None, top=None, **kwargs):
        '''
        Featurize the trajectories in rtraj_dir and save the feature trajectories to ftraj_dir.

        Parameters
        ----------
        key : str
            The key to the dataset
        featurisers : list
            A list of functions that featurize the trajectories
        feature_names : list or None
            A list of names for saving featurised trajectories. If None, the names will be the same as the featurisers
        top : str or None
            The topology file for the trajectories. If None, assume the trajectory format include topology information
        kwargs : dict
            Some featuring functions may require additional arguments
        '''

        try:
            save_dir = self.datasets[key]['ftraj_dir']
            traj_files = self.datasets[key]['rtraj_files']
        except KeyError:
            raise ValueError(f'Dataset {key} not found. Set the dataset first.')

        if not save_dir.exists(): save_dir.mkdir(parents=True, exist_ok=True)
        if feature_names is not None: assert len(featurisers) == len(feature_names), 'The number of featurisers and feature names do not match'

        print(f'Featurising protein {self.protein}')
        print(f'Featurisers: {[f.__name__ for f in featurisers]}')

        if feature_names is None:
            names = [featuriser.__name__.split('_')[0] for featuriser in featurisers]
        else:
            names = feature_names
        
        for name in names:
            if not (save_dir / name).exists(): 
                (save_dir / name).mkdir(parents=True, exist_ok=True)

        for i, traj in tqdm(enumerate(traj_files), total=len(traj_files)):
            md_traj = None
            print('Featurising', traj.stem)

            for featuriser, name in zip(featurisers, names):
                
                ftraj_f = save_dir / name / f"{traj.stem}_{name}.npy"
                # ftraj_f = save_dir / name / f"{traj.stem}.npy"
                if ftraj_f.is_file():
                    print(traj.stem, f"{name} ftraj already exist.")
                    continue

                if md_traj is None:
                    try:
                        if top is None:
                            md_traj = md.load(traj)
                        else:
                            md_traj = md.load(traj, top=top)
                    except:
                        print(f'!!! Fail to read {traj} !!!')
                        break

                try:
                    _ = featuriser(md_traj, self.protein, save_to_disk=ftraj_f, **kwargs)
                    print('Featurised', traj.stem, f"{name} ftraj.")
                except Exception as err:
                    if os.path.exists(ftraj_f): os.remove(ftraj_f)
                    print(f"Fail to featurise {traj.stem} with {name}:\n{err}")


    def load_ftrajs(self, key, feature_names, internal_names=None):
        '''
        Load the featurised trajectories from the save directory.
        
        Parameters
        ----------
        key : str
            The key to the dataset
        feature_names : list
            A list of strings of features. Should correspond to the subdirectory of ftraj files in the ftraj directory
        internal_names : list or None
            A list of names to assign to the ftrajs loaded internally. If None, the names will be the same as the features
        '''

        try:
            print(key)
            save_dir = self.datasets[key]['ftraj_dir']
        except KeyError:
            raise ValueError(f'Dataset {key} not found. Set the dataset first.')

        if internal_names is not None: 
            assert len(feature_names) == len(internal_names), 'The number of features and feature names do not match'
            names = internal_names
        else:
            names = feature_names
        
        new_ftraj_files_ls = []
        new_feature_ls = []
        new_name_ls = []

        for feature, name in zip(feature_names, names):
            if name in self._ftrajs and self._ftrajs[name] is not None:
                print(f'Feature {feature} already loaded as {name}. Skipping...')
            else:
                ftraj_files = natsorted([str(ftraj) for ftraj in (save_dir/name).glob(f'*_{feature}.npy')])
                if len(ftraj_files) == 0:
                    raise ValueError(f'No feature trajectories found for {feature}. Check the directory.')
                new_ftraj_files_ls.append(ftraj_files)
                new_feature_ls.append(feature)
                new_name_ls.append(name)

        ftraj_files_lens = [len(traj_files) for traj_files in new_ftraj_files_ls]
        if not all(len == ftraj_files_lens[0] for len in ftraj_files_lens):
            raise ValueError('Feature trajectories are not of equal length. Check if the feature trajectories exist.')

        for ftraj_files, feature, name in zip(new_ftraj_files_ls, new_feature_ls, new_name_ls):
            print("Loading feature: ", feature)
            feature_ls = []
            for ftraj_file in tqdm(ftraj_files, total=len(ftraj_files)):
                ftraj = np.load(ftraj_file, allow_pickle=True)
                if ftraj.ndim == 1:
                    ftraj = ftraj[:, np.newaxis]
                feature_ls.append(ftraj)
            self._ftrajs[key][name] = feature_ls

    
    @staticmethod
    def prepare_ftrajs(ftraj_dict, stride, frame_no_cutoff, convert_dihed_ids):
        '''
        Concatenate and prepare the feature trajectories for the MSM analysis. 

        Parameters
        ----------
        ftraj_dict : dict
            A dictionary of the feature trajectories to prepare
        stride : int
            The stride to sample the feature trajectories
        frame_no_cutoff : int
            The minimum no of frames in feature trajectories for them to be included
        convert_dihed_ids : list or None
            A list of indices of the dihedral features to convert to cos and sin. If None, no conversion is done

        Returns
        -------
        ftrajs : list
            A list of concatenated feature trajectories
        mapping : dict
            A dictionary mapping the indices of the selected feature trajectories to the original feature trajectories
        '''
        
        ftrajs_raw = [v for v in ftraj_dict.values()]
        no_of_trajs = len(ftrajs_raw[0])

        ftrajs, mapping = [], {}
        print("Preparing feature trajectories...")
        for i in tqdm(range(no_of_trajs), total=no_of_trajs):
            if ftrajs_raw[0][i].shape[0] < frame_no_cutoff:
                continue

            ftrajs_to_add = [ftraj[i] for ftraj in ftrajs_raw]
            if convert_dihed_ids is not None:
                for id in convert_dihed_ids:
                    ftrajs_to_add[id] = np.concatenate([np.cos(ftrajs_to_add[id]), np.sin(ftrajs_to_add[id])], axis=1)
            
            if ftrajs_to_add[0].ndim == 1:
                ftrajs_to_add = np.concatenate(ftrajs_to_add)[::stride]
            else:
                ftrajs_to_add = np.concatenate(ftrajs_to_add, axis=1)[::stride, :]
                
            ftrajs.append(ftrajs_to_add)
            mapping[len(ftrajs)-1] = i

        return ftrajs, mapping 


    def get_ftrajs(self, keys, dt_out, internal_names, time_cutoff=0, convert_dihed_ids=None):
        '''
        Get the feature trajectories for the internal names provided.

        Parameters
        ----------
        keys : str or list
            The dataset key to get the feature trajectories from. If a list, the feature trajectories of datasets are concatenated
        dt_out : float
            The output time step of the feature trajectories in ns. Should be a multiple of the shortest time step in the datasets
        internal_names : list
            A list of names in the internal ftrajs dictionary to prepare
        time_cutoff : int
            The minimum time length of the feature trajectories to be considered in ns
        convert_dihed_ids : list or None
            A list of indices of the dihedral features to convert to cos and sin. If None, no conversion is done
        
        Returns
        -------
        ftrajs : list
            A list of concatenated feature trajectories
        mappings : dict
            A dictionary mapping the indices of the selected feature trajectories to the original feature trajectories
        '''
        
        if isinstance(keys, str):
            keys = [keys]
            
        for key in keys:
            if not key in self.datasets.keys():
                raise KeyError(f'Dataset key {key} not found. Set the dataset first.')
            if not all(name in self._ftrajs[key].keys() for name in internal_names):
                raise ValueError(f'Not all internal names provided are loaded for key {key}. Load the feature trajectories first.')

        ftrajs, mapping, last_ftraj_no, last_rtraj_no = [], {}, 0, 0
        for key in keys:
            ftraj_dict = {name:self._ftrajs[key][name] for name in internal_names}
            ftraj_dt = self.datasets[key]['dt']
            stride = int(dt_out / ftraj_dt)
            frame_no_cutoff = int(time_cutoff / ftraj_dt)
            print(f"Stride for dataset {key} with timestep {self.datasets[key]['dt']} ns: {stride}")
            
            f, m = self.prepare_ftrajs(ftraj_dict, stride=stride, frame_no_cutoff=frame_no_cutoff, convert_dihed_ids=convert_dihed_ids)
            m = {k+last_ftraj_no:v+last_rtraj_no for k, v in m.items()}
            last_ftraj_no += len(f)
            last_rtraj_no += len(self.datasets[key]['rtraj_files'])
            mapping.update(m)
            ftrajs.extend(f)

        return ftrajs, mapping

    
    def del_ftrajs(self, key, feature_names):
        '''
        Delete the feature trajectories from the internal ftrajs dictionary.

        Parameters
        ----------
        key : str
            The key to the dataset
        feature_names : list
            A list of names of the feature trajectories to delete
        '''

        for name in feature_names:
            if name in self._ftrajs[key]:
                del self._ftrajs[key][name]
                print(f'Deleted ftrajs <{name}> from dataset <{key}>')
            else:
                print(f'{name} not found in {key}')


    @property
    def ftrajs_keys(self):
        '''
        Get the keys of the internal ftrajs dictionary.
        '''
        for key in self._ftrajs.keys():
            print(f'Dataset <{key}> ftrajs: {list(self._ftrajs[key].keys())}')

