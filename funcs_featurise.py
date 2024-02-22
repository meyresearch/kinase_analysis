import mdtraj as md
import numpy as np
from typing import *


def dunbrack_egfr_featurizer(traj) -> np.ndarray:
    # The dunbrack featurizer for the EGFR kinase domain

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


def featurize(featurizers: List[function], traj: md.Trajectory) -> np.ndarray:
    assert type(traj) == md.Trajectory
    
    ftraj_list = []
    for featurizer in featurizers:
        ftraj = featurizer(traj)
        if ftraj.dim == 1: 
            ftraj = ftraj[:, np.newaxis]
        ftraj_list.append(ftraj)

    return np.concatenate(ftraj_list, axis=1)
