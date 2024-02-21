import mdtraj as md
import numpy as np

import re
import numpy
from tqdm import tqdm
from pathlib import Path
import json


def featurizer(traj, subset_atomids, subset_caids):
    # Ca distance features
    f_dist = md.compute_distances(traj, [(i, j) for i in subset_caids for j in subset_caids if i < j])
    
    # Dihedral features 
    phi_indices, phi = md.compute_phi(traj)
    psi_indices, psi = md.compute_psi(traj)
    chi1_indices, chi1 = md.compute_chi1(traj)
    f_phi = phi[:, np.any(np.isin(phi_indices, subset_atomids), axis=1)]
    f_psi = psi[:, np.any(np.isin(psi_indices, subset_atomids), axis=1)]
    f_chi1 = chi1[:, np.any(np.isin(chi1_indices, subset_atomids), axis=1)]
    f_dihed = np.concatenate([f_phi, f_psi, f_chi1], axis=1)
    f_dihed = np.concatenate([np.cos(f_dihed), np.sin(f_dihed)], axis=1)
    
    # Ca coordinates 
    f_ca = traj.xyz[:, subset_caids, :]    
    return f_dist, f_dihed, f_ca


# Select the subset of residues that will be featurized

# -----------------------Abl1--------------------------
traj_dir = Path('/home/rzhu/stoarge_SRS/trajectories/aws/')
save_dir = Path('/home/rzhu/Desktop/projects/kinase_analysis/ftraj_abl1/')
files = [f for f in traj_dir.iterdir() if re.match(r'run[0-9]+-clone[0-9]+\.h5', f.name)]
max_value = max([int(re.search(r'run([0-9]+)-clone[0-9]+\.h5', f.name).group(1)) for f in files])
ref = md.load('/home/rzhu/Desktop/projects/kinase_analysis/human_abl1/kinoml_OEKLIFSKinaseApoFeaturizer_ABL1_1opl_chainA_altlocNone_protein.pdb')

with open('./human_abl1/uniprot_resids_to_klifs_resids.json','r') as f:
    mapping = json.load(f)
klifs_pocket_resids = [int(k) for k,v in mapping.items()]
p_loop_resids = list(range(249, 255))
c_helix_resids = list(range(282, 293))
active_loop_resids = list(range(381, 403))
'''

# -----------------------EGFR--------------------------
traj_dir = Path('/home/rzhu/stoarge_SRS/trajectories/aws2/')
save_dir = Path('/home/rzhu/Desktop/projects/kinase_analysis/ftraj_egfr/')
files = [f for f in traj_dir.iterdir() if re.match(r'run[0-9]+-clone[0-9]+\.h5', f.name)]
max_value = max([int(re.search(r'run([0-9]+)-clone[0-9]+\.h5', f.name).group(1)) for f in files])
ref = md.load('/home/rzhu/Desktop/projects/kinase_analysis/human_egfr/kinoml_OEKLIFSKinaseApoFeaturizer_EGFR_1m14_chainA_altlocNone_protein.pdb')

with open('./human_egfr/uniprot_resids_to_klifs_resids.json','r') as f:
    mapping = json.load(f)
klifs_pocket_resids = [int(k) for k,v in mapping.items()]
p_loop_resids = list(range(695, 701))
c_helix_resids = list(range(734, 746))
active_loop_resids = list(range(831, 855))
'''
backbone_atomids = ref.topology.select('backbone')
subset_resids = np.unique(klifs_pocket_resids + active_loop_resids + c_helix_resids + p_loop_resids)
subset_atomids = ref.topology.select('residue ' + ' '.join(map(str, subset_resids)))
subset_caids = ref.topology.select('name CA and residue ' + ' '.join(map(str, subset_resids)))
print('No of subset residues:', len(subset_resids),\
      '\nNo of subset atoms:', len(subset_atomids),\
      '\nNo of subset Ca:', len(subset_caids))
# Note that EGFRs don't have all the p loop residues, so no subset residues != no subset ca

# Loop over each starting conformation 
for i in tqdm(range(max_value), total=max_value):
    trajs = [p for p in traj_dir.rglob(f'run{i}-clone?.h5')]
    
    # Loop over each repetition 
    for traj in trajs:
        if save_dir.joinpath(f'{traj.stem}_dist.npy').is_file() and \
           save_dir.joinpath(f'{traj.stem}_dihed.npy').is_file() and \
           save_dir.joinpath(f'{traj.stem}_ca.npy').is_file():
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
        f_dist, f_dihed, f_ca = featurizer(md_traj, subset_atomids, subset_caids)

        with open(save_dir.joinpath(f'{traj.stem}_dist.npy'),'wb') as f:
            np.save(f, f_dist)        
        with open(save_dir.joinpath(f'{traj.stem}_dihed.npy'),'wb') as f:
            np.save(f, f_dihed)           
        with open(save_dir.joinpath(f'{traj.stem}_ca.npy'),'wb') as f:
            np.save(f, f_ca)
