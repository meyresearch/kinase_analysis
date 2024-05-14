import re
import os
from natsort import natsorted
from tqdm import tqdm
from pathlib import Path
import builtins
from funcs_abl_featurise import *

original_print = builtins.print
def print(*args, **kwargs):
    kwargs.setdefault('flush', True)  # Set flush to True unless it's already specified
    original_print(*args, **kwargs)
builtins.print = print


if __name__ == "__main__":
    traj_dir = Path('/arc/abl_processed')
    traj_files = natsorted([traj for traj in traj_dir.rglob('run*-clone?.h5')])
    max_run_no = max([int(re.search(r'run([0-9]+)-clone[0-9]+\.h5', f.name).group(1)) for f in traj_files])

    save_dir = Path('/home/rzhu/Desktop/projects/kinase_analysis/data_abl/ftrajs/')
    if not save_dir.exists(): save_dir.mkdir(parents=True, exist_ok=True)

    # No need to align to a reference structure as all the features are internal degrees of freedoms
    # ref = md.load('/arc/egfr_equilibrated_strucs/RUN0_solute_equilibrated.pdb')
    
    featurisers = [ploopdihed_featuriser]
    print(f'Featurisers: {[f.__name__ for f in featurisers]}')

    # Loop over runs
    try:
        for i in tqdm(range(max_run_no+1), total=max_run_no+1):
            trajs = natsorted([p for p in traj_dir.rglob(f'run{i}-clone?.h5')])
            print(f'Run {i}: {len(trajs)} trajectories to be featurised.')

            # Loop over clones
            for traj in trajs:
                md_traj = None
                print('Featurising', traj.stem)

                for featuriser in featurisers:
                    ftraj_dir = save_dir.joinpath(f"{traj.stem}_{featuriser.__name__.split('_')[0]}.npy")
                    if ftraj_dir.is_file():
                        print(traj.stem, f"{featuriser.__name__.split('_')[0]} ftraj already exist.")
                        continue

                    if md_traj is None:
                        try:
                            md_traj = md.load(traj)
                            # md_traj = md_traj.superpose(ref, atom_indices=md_traj.topology.select('backbone'), ref_atom_indices=ref.topology.select('backbone'))
                        except:
                            print(f'!!! Fail to read {traj} !!!')
                            break

                    try:
                        _ = featuriser(md_traj, save_to_disk=ftraj_dir)
                        print('Featurised', traj.stem, f"{featuriser.__name__.split('_')[0]} ftraj.")
                    except Exception as err:
                        if os.path.exists(ftraj_dir): os.remove(ftraj_dir)
                        print(f"Fail to featurise {traj.stem} with {featuriser.__name__.split('_')[0]}:\n{err}")
    except KeyboardInterrupt:
        print('Keyboard interrupt. Exiting.')