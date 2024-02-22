import re
from natsort import natsorted
from tqdm import tqdm
from pathlib import Path

from funcs_featurise import *


if __name__ == "__main__":
    traj_dir = Path('/arc/EGFR/')
    traj_files = natsorted([traj for traj in traj_dir.rglob('run*-clone?.h5')])
    max_run_no = max([int(re.search(r'run([0-9]+)-clone[0-9]+\.h5', f.name).group(1)) for f in traj_files])

    save_dir = Path('/home/rzhu/Desktop/projects/kinase_analysis/data_egfr/ftrajs/')
    if not save_dir.exists(): save_dir.mkdir(parents=True, exist_ok=True)

    ref = md.load('/home/rzhu/Desktop/projects/kinase_analysis/human_egfr/kinoml_OEKLIFSKinaseApoFeaturizer_EGFR_1m14_chainA_altlocNone_protein.pdb')
    backbone_atomids = ref.topology.select('backbone')

    # Loop over each starting conformation 
    for i in tqdm(range(max_run_no), total=max_run_no):
        trajs = [p for p in traj_dir.rglob(f'run{i}-clone?.h5')]
        
        # Loop over each repetition 
        for traj in trajs:
            if save_dir.joinpath(f'{traj.stem}_dunbrack.npy').is_file():
                print(traj, 'Already exist.')
                continue
            try:
                md_traj = md.load(traj)
            except:
                with open(save_dir.joinpath('log.txt'), 'a') as f:
                    f.writelines(f'Fail to read {traj} \n')
                print(f'Fail to read {traj} \n')
                continue
                
            print(traj, len(md_traj))
            md_traj = md_traj.superpose(ref, atom_indices=backbone_atomids)

            ftraj = featurize([dunbrack_egfr_featurizer], md_traj)
            with open(save_dir.joinpath(f'{traj.stem}_dunbrack.npy'),'wb') as f:
                np.save(f, ftraj)