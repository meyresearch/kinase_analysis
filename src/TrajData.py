from pathlib import Path
from natsort import natsorted
from tqdm import tqdm
import numpy as np
import os
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
        self.datasets = {}
        self._ftrajs = {}
    

    def add_dataset(self, key, rtraj_dir, ftraj_dir, dt, rtraj_glob='*', exclude_files=None):
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
        rtraj_glob : str
            Glob pattern to search for the raw trajectory files. By default all files in the directory are considered as trajectories.
        exclude_files : list of str, optional
            List of exact filenames to exclude from the dataset
        '''

        if key in self.datasets:
            print(f'Key <{key}> already exists in the dataset.')
            return None

        rtraj_dir = Path(rtraj_dir)
        ftraj_dir = Path(ftraj_dir)
        rtraj_files = natsorted(rtraj_dir.glob(rtraj_glob))

        if not rtraj_files:
            raise ValueError(f'No raw trajectory files found in {rtraj_dir}. Check the directory or glob pattern.')

        if exclude_files:
            exclude = set(exclude_files)
            filtered = [traj for traj in rtraj_files if traj.name not in exclude]
            if len(filtered) != len(rtraj_files):
                removed = len(rtraj_files) - len(filtered)
                print(f'Excluded {removed} trajectories: {sorted(exclude)}')
            rtraj_files = filtered

        print(f'Setting dataset <{key}>. \nNumber of raw trajectories: {len(rtraj_files)}\n')
        self.datasets[key] = {
            'rtraj_dir': rtraj_dir,
            'ftraj_dir': ftraj_dir,
            'rtraj_files': rtraj_files,
            'dt': dt,
        }
        self._ftrajs[key] = {}


    def __str__(self):
        '''Represent the object with the protein name and registered datasets.''' 
        return f'TrajData for: {self.protein}\nDatasets: {list(self.datasets.keys())}'


    def featurize(self, key, featurisers, feature_names=None, top=None, **kwargs):
        '''
        Run the selected featurisers on every raw trajectory in ``key`` and store them on disk.

        Parameters
        ----------
        key : str
            Dataset identifier registered with :meth:`add_dataset`.
        featurisers : Iterable[Callable]
            Callables that accept an ``mdtraj.Trajectory`` and persist the computed features.
        feature_names : Iterable[str], optional
            Custom directory/filename prefixes for each featuriser. Defaults to the featuriser
            name prefix when omitted.
        top : str or pathlib.Path, optional
            Optional topology passed through to :func:`mdtraj.load`.
        **kwargs : dict
            Extra keyword arguments forwarded to each featuriser.
        '''

        dataset = self._get_dataset(key)
        save_dir = dataset['ftraj_dir']
        traj_files = dataset['rtraj_files']

        save_dir.mkdir(parents=True, exist_ok=True)
        if feature_names is not None and len(featurisers) != len(feature_names):
            raise ValueError('The number of featurisers and feature names do not match')

        names = feature_names or [f.__name__.split('_')[0] for f in featurisers]
        for name in names:
            (save_dir / name).mkdir(parents=True, exist_ok=True)

        print(f'Featurising protein {self.protein}')
        print(f'Featurisers: {[f.__name__ for f in featurisers]}')

        for traj in tqdm(traj_files, total=len(traj_files)):
            md_traj = None
            for featuriser, name in zip(featurisers, names):
                ftraj_path = save_dir / name / f"{traj.stem}_{name}.npy"
                if ftraj_path.exists():
                    print(f'Feature trajectory {ftraj_path} already exists. Skipping.')
                    continue

                if md_traj is None:
                    try:
                        md_traj = md.load(traj, top=top) if top else md.load(traj)
                    except Exception as err:
                        print(f'Failed to read {traj}: {err}')
                        break

                try:
                    featuriser(md_traj, self.protein, save_to_disk=ftraj_path, **kwargs)
                except Exception as err:
                    if ftraj_path.exists():
                        os.remove(ftraj_path)
                    print(f"Fail to featurise {traj.stem} with {name}: {err}")


    def load_ftrajs(self, key, feature_names, internal_names=None):
        '''
        Load feature trajectories from disk into memory for the requested dataset.

        Parameters
        ----------
        key : str
            Dataset identifier registered with :meth:`add_dataset`.
        feature_names : Iterable[str]
            Directory names corresponding to previously featurised trajectories.
        internal_names : Iterable[str], optional
            Alternative keys to store the loaded trajectories under; defaults to
            ``feature_names`` when omitted.
        '''

        dataset = self._get_dataset(key)
        save_dir = dataset['ftraj_dir']
        rtraj_files = dataset['rtraj_files']

        names = internal_names or feature_names
        if len(feature_names) != len(names):
            raise ValueError('The number of features and internal names do not match')

        cache = self._ftrajs.setdefault(key, {})

        for feature, name in zip(feature_names, names):
            if cache.get(name) is not None:
                continue

            expected = [save_dir / name / f"{traj.stem}_{feature}.npy" for traj in rtraj_files]
            missing = [path for path in expected if not path.exists()]
            if missing:
                raise ValueError(f'Feature trajectories missing for {feature}: {missing[:3]}')

            trajs = []
            for ftraj_path in tqdm(expected, total=len(expected), desc=f'Loading {feature} ({key})'):
                ftraj = np.load(ftraj_path, allow_pickle=True)
                if ftraj.ndim == 1:
                    ftraj = ftraj[:, np.newaxis]
                trajs.append(ftraj)

            cache[name] = trajs
    

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
            The output time interval of the feature trajectories in ns. Should be a multiple of the shortest time step in the datasets
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


    def _get_dataset(self, key):
        '''Return the dataset metadata for ``key`` or raise a helpful error.''' 
        try:
            return self.datasets[key]
        except KeyError as exc:
            raise ValueError(f'Dataset {key} not found. Set the dataset first.') from exc