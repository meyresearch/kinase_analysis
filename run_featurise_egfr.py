import mdtraj as md
import numpy as np

import re
import numpy
from tqdm import tqdm
from pathlib import Path
import json


def featurizer(traj):
    """
    Compute features for kinase analysis.

    Args:
        traj (mdtraj.Trajectory): The input trajectory.

    Returns:
        np.ndarray: The computed features.

    Raises:
        AssertionError: If the atom IDs do not match the expected values.
    """
    
    ########## Spatial groups ############
    # d1 = dist(αC-Glu(+4)-Cα, DFG-Phe-Cζ) 
    # Distance between the Ca of the fourth residue after the conserved Glu (Met) in the C-helix
    # and the outermost atom of the DFG-Phe ring Cζ
    # d2 = dist(β3-Lys-Cα, DFG-Phe-Cζ) 
    ######################################
    
    met_ca_id = 1026
    phe_cz_id = 2496
    lys_ca_id = 687
    
    top = traj.topology
    assert str(top.atom(met_ca_id)) == 'MET766-CA'
    assert str(top.atom(phe_cz_id)) == 'PHE856-CZ'
    assert str(top.atom(lys_ca_id)) == 'LYS745-CA'
    d = md.compute_distances(traj, [[met_ca_id, phe_cz_id], [lys_ca_id, phe_cz_id]])
    
    ########## Dihedral groups ############
    # The backbone dihedrals of 
    # X-DFG (residue before conserved Asp, Thr), DFG-Asp, DFG-Phe;
    # and the side chain dihedral (χ1) of DFG-Phe
    ######################################
    
    idx_psi, psi = md.compute_psi(traj)
    idx_phi, phi = md.compute_phi(traj)
    idx_chi, chi = md.compute_chi1(traj)

    phi_f = phi[:, 150:153]
    psi_f = psi[:, 150:153]
    phe_chi1 = chi[:, 134]
    
    assert (idx_phi[150] == [2443, 2460, 2461, 2462]).all()
    assert (idx_psi[150] == [2460, 2461, 2462, 2474]).all()
    assert (idx_chi[134] == [2486, 2487, 2490, 2491]).all()
    
    # Convert angles to sin and cos
    cos = np.cos(np.concatenate([phi_f, psi_f, phe_chi1[:, np.newaxis]], axis=1))
    sin = np.sin(np.concatenate([phi_f, psi_f, phe_chi1[:, np.newaxis]], axis=1))

    return np.concatenate([d, cos, sin], axis=1)



if __name__ == "__main__":
    traj_dir = Path('/home/rzhu/stoarge_SRS/trajectories/aws2/')
    save_dir = Path('/home/rzhu/Desktop/projects/kinase_analysis/ftraj_egfr/')
    files = [f for f in traj_dir.iterdir() if re.match(r'run[0-9]+-clone[0-9]+\.h5', f.name)]
    max_value = max([int(re.search(r'run([0-9]+)-clone[0-9]+\.h5', f.name).group(1)) for f in files])
    ref = md.load('/home/rzhu/Desktop/projects/kinase_analysis/human_egfr/kinoml_OEKLIFSKinaseApoFeaturizer_EGFR_1m14_chainA_altlocNone_protein.pdb')
    backbone_atomids = ref.topology.select('backbone')

    # Loop over each starting conformation 
    for i in tqdm(range(max_value), total=max_value):
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
            f_dunbrack = featurizer(md_traj)

            with open(save_dir.joinpath(f'{traj.stem}_dunbrack.npy'),'wb') as f:
                np.save(f, f_dunbrack)
