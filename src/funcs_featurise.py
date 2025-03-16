import mdtraj as md
import numpy as np
from typing import *
from funcs_indices import *


################## Features from Dunbrack paper ####################

def dbdist_featuriser(traj, protein, save_to_disk=None) -> np.ndarray:

    '''
    Parameters
    ----------
    traj : mdtraj.Trajectory
        The trajectory object of a simulation
    protein : str
        The name of the protein in the topology to get relevant atom indices
    save_to_disk : str
        The path to save the feature to disk
    
    Returns
    -------
    np.ndarray
        A feature vector
    '''

    ########## Dunbrack DFG distances ############
    # d1 = dist(αC-Glu(+4)-Cα, DFG-Phe-Cζ) 
    # Distance between the Ca of the fourth residue after the conserved Glu (Met) in the C-helix
    # and the outermost atom of the DFG-Phe ring Cζ
    # d2 = dist(β3-Lys-Cα, DFG-Phe-Cζ) 
    ##############################################

    indices = get_feature_indices(traj.topology, protein, 'db_dist')
    distances = md.compute_distances(traj, [[indices['aC_M-ca'], indices['dfg_F-cz']], 
                                            [indices['b3_K-ca'], indices['dfg_F-cz']]])
    if save_to_disk is not None: np.save(save_to_disk, distances)
    return distances


def dbdihed_featuriser(traj, protein, save_to_disk=None) -> np.ndarray:
    ########## Dunbrack DFG dihedrals ############
    # The backbone dihedrals of 
    # X-DFG (residue before conserved Asp, Thr), DFG-Asp, DFG-Phe;
    # and the side chain dihedral (χ1) of DFG-Phe
    ##############################################
    indices = get_feature_indices(traj.topology, protein, 'db_dihed')
    dihedrals = md.compute_dihedrals(traj, 
                                     [[indices['XX_dfg-c'], indices['X_dfg-n'], indices['X_dfg-ca'], indices['X_dfg-c']], # X-DFG phi
                                      [indices['X_dfg-n'], indices['X_dfg-ca'], indices['X_dfg-c'], indices['dfg_D-n']],  # X-DFG psi
                                      [indices['X_dfg-c'], indices['dfg_D-n'], indices['dfg_D-ca'], indices['dfg_D-c']],  # DFG-D phi
                                      [indices['dfg_D-n'], indices['dfg_D-ca'], indices['dfg_D-c'], indices['dfg_F-n']],  # DFG-D psi
                                      [indices['dfg_D-c'], indices['dfg_F-n'], indices['dfg_F-ca'], indices['dfg_F-c']],  # DFG-F phi
                                      [indices['dfg_F-n'], indices['dfg_F-ca'], indices['dfg_F-c'], indices['dfg_G-n']],  # DFG-F psi
                                      [indices['dfg_F-c'], indices['dfg_G-n'], indices['dfg_G-ca'], indices['dfg_G-c']],  # DFG-G phi (unimportant)
                                      [indices['dfg_G-n'], indices['dfg_G-ca'], indices['dfg_G-c'], indices['dfg_X-n']],  # DFG-G psi (unimportant)
                                      [indices['dfg_F-n'], indices['dfg_F-ca'], indices['dfg_F-cb'], indices['dfg_F-cg']]]) # DFG-F chi1
    if save_to_disk is not None: np.save(save_to_disk, dihedrals)
    return dihedrals


def aloop_featuriser(traj, protein, save_to_disk=None) -> np.ndarray:
    ########## Activation loop contacts ############
    # Pairwise distances between Cα atoms of residue i and i+3
    # Activation loop starts from ASP381 and ends at ILE403
    ################################################
    indices = get_feature_indices(traj.topology, protein, 'aloop')
    aloop_traj = traj.atom_slice(traj.topology.select(f'resid {indices["aloop_start"]} to {indices["aloop_end"]}'))
    distances = md.compute_contacts(aloop_traj, contacts='all', scheme='ca')[0]
    if save_to_disk is not None: np.save(save_to_disk, distances)
    return distances


def achelix_featuriser(traj, protein, save_to_disk=None) -> np.ndarray:
    ########## phosphate-binding loop distances ############
    # d1 = dist(Nz-Lys271, Cd-Glu286) 
    # d2 = dist(Cz-Arg386, Cd-Glu286) 
    ########################################################
    indices = get_feature_indices(traj.topology, protein, 'aChelix_dist')
    distances = md.compute_distances(traj, [[indices['b3_K-nz'], indices['aC_E-cd']], 
                                            [indices['aloop_R/K-cz'], indices['aC_E-cd']]])
    if save_to_disk is not None: np.save(save_to_disk, distances)
    return distances


def aloopdssp_featuriser(traj, protein, save_to_disk=None) -> np.ndarray:
    ########## Activation loop secondary structure ############
    # Secondary structure assignment of activation loop
    ###########################################################
    indices = get_feature_indices(traj.topology, protein, 'aloop')
    aloop_traj = traj.atom_slice(traj.topology.select(f'resid {indices["aloop_start"]} to {indices["aloop_end"]}'))
    dssp = md.compute_dssp(aloop_traj)
    #fraction_of_helix = np.mean(dssp == 'H', axis=1)
    if save_to_disk is not None: np.save(save_to_disk, dssp)
    return dssp


def achelixdssp_featuriser(traj, protein, save_to_disk=None) -> np.ndarray:
    ########## alpha-C helix secondary structure ################
    # Secondary structure assignment of alpha-C helix
    # Alpha-C helix is defined as 
    # GLU281 to GLU292
    #############################################################
    indices = get_feature_indices(traj.topology, protein, 'aChelix')
    achelix_traj = traj.atom_slice(traj.topology.select(f'resid {indices["aC_start"]} to {indices["aC_end"]}'))   
    dssp = md.compute_dssp(achelix_traj)
    #fraction_of_helix = np.mean(dssp == 'H', axis=1)
    if save_to_disk is not None: np.save(save_to_disk, dssp)
    return dssp


def achelixdihed_featuriser(traj, protein, save_to_disk=None) -> np.ndarray:
    ########## alpha-C helix secondary structure ################
    # Dihedral angles of alpha-C helix
    # Alpha-C helix is defined as 
    # GLU281 to GLU292
    #############################################################
    indices = get_feature_indices(traj.topology, protein, 'aChelix')
    achelix_traj = traj.atom_slice(traj.topology.select(f'resid {indices["aC_start"]} to {indices["aC_end"]}'))   
    phi = md.compute_phi(achelix_traj)[1]
    psi = md.compute_psi(achelix_traj)[1]
    dihedrals = np.concatenate([phi, psi], axis=1)
    if save_to_disk is not None: np.save(save_to_disk, dihedrals)
    return dihedrals


################## Features from the mechanism paper ####################

def inHbonddist_featuriser(traj, protein, save_to_disk=None) -> np.ndarray:
    indices = get_feature_indices(traj.topology, protein, 'inHbond_dist')
    distance = md.compute_distances(traj, [[indices['dfg_D-cg'], indices['ploop_Hdonor']]])
    if save_to_disk is not None: np.save(save_to_disk, distance)
    return distance


def interHbond1dist_featuriser(traj, protein, save_to_disk=None) -> np.ndarray:
    indices = get_feature_indices(traj.topology, protein, 'interHbond1_dist')
    distance = md.compute_distances(traj, [[indices['dfg_D-cg'], indices['b5_T-og1']]])
    if save_to_disk is not None: np.save(save_to_disk, distance)
    return distance


def interHbond2dist_featuriser(traj, protein, save_to_disk=None) -> np.ndarray:
    indices = get_feature_indices(traj.topology, protein, 'interHbond2_dist')
    distance = md.compute_distances(traj, [[indices['dfg_D-cg'], indices['b4_backbone']]])
    if save_to_disk is not None: np.save(save_to_disk, distance)
    return distance


def interpipidist_featuriser(traj, protein, save_to_disk=None) -> np.ndarray:
    indices = get_feature_indices(traj.topology, protein, 'interpipi_dist')
    distance = md.compute_distances(traj, [[indices['dfg_F-cg'], indices['hrd_H-ne2']]])
    if save_to_disk is not None: np.save(save_to_disk, distance)
    return distance


def outpipidist_featuriser(traj, protein, save_to_disk=None) -> np.ndarray:
    indices = get_feature_indices(traj.topology, protein, 'outpipi_dist')
    distance = md.compute_distances(traj, [[indices['dfg_F-cg'], indices['ploop_ring-cg']]])
    if save_to_disk is not None: np.save(save_to_disk, distance)
    return distance


def pathwayangle_featuriser(traj, protein, save_to_disk=None) -> np.ndarray:
    indices = get_feature_indices(traj.topology, protein, 'pathway_angle')
    angle = md.compute_angles(traj, [[indices['dfg_F-cg'], indices['dfg_D-cg'], indices['(D-3)_K-O']]])
    if save_to_disk is not None: np.save(save_to_disk, angle)
    return angle


def alooprmsd_featuriser(traj, protein, reference, save_to_disk=None) -> np.ndarray:
    indices = get_feature_indices(traj.topology, protein, 'aloop')
    traj = traj.superpose(reference, atom_indices=traj.topology.select('name CA'))
    rmsd = md.rmsd(traj, reference, atom_indices=traj.topology.select(f'resid {indices["aloop_start"]} to {indices["aloop_end"]}'))
    if save_to_disk is not None: np.save(save_to_disk, rmsd)
    return rmsd 


## The writhe featurizer 

import wiggle.writhe as wr

def get_sigma_tensor(coords):
    # get the shape of the tensor - every 4th CA step
    tmp = wr.find_Sigma_array(coords[0][::4])
    # initialise array
    sigma_tensor = np.zeros((coords.shape[0], tmp.shape[0], tmp.shape[0]))
    for i in range(coords.shape[0]):
        sigma_tensor[i] = wr.find_Sigma_array(coords[i][::4])
    return sigma_tensor

def writhe_featuriser(traj, protein, flatten=True, save_to_disk=None) -> np.ndarray:
    '''
    Parameters
    ----------
    traj : mdtraj.Trajectory
        The trajectory object of a simulation
    protein : str
        The name of the protein in the topology to get relevant atom indices
    save_to_disk : str
        The path to save the feature to disk
    Returns
    -------
    np.ndarray
        A feature vector - tensor of shape (n_frames, 26, 26)
    '''

    ca_indices = traj.topology.select('name CA')
    ca_coords = traj.xyz[:, ca_indices] # im assuming this loads a (number_frames, number_residues, 3) array
    sigma_tensor = get_sigma_tensor(ca_coords)
    if flatten:
        sigma_tensor = sigma_tensor.reshape(sigma_tensor.shape[0], -1)
    if save_to_disk is not None:
        np.save(save_to_disk, sigma_tensor)
    return sigma_tensor

