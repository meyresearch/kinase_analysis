import mdtraj as md
import numpy as np
from typing import *
from scipy.spatial.distance import cdist


def find_atomid(top, atom_name) -> int:
    atomid = np.where([str(atom) == atom_name for atom in top.atoms])[0]
    assert len(atomid) == 1

    return atomid[0]


def dbdist_featuriser(traj, save_to_disk=None) -> np.ndarray:
    ########## Dunbrack DFG distances ############
    # d1 = dist(αC-Glu(+4)-Cα, DFG-Phe-Cζ) 
    # Distance between the Ca of the fourth residue after the conserved Glu (Met) in the C-helix
    # and the outermost atom of the DFG-Phe ring Cζ
    # d2 = dist(β3-Lys-Cα, DFG-Phe-Cζ) 
    ##############################################
    top = traj.topology

    met_ca_id = find_atomid(top, 'MET766-CA')
    phe_cz_id = find_atomid(top, 'PHE856-CZ')
    lys_ca_id = find_atomid(top, 'LYS745-CA')

    distances = md.compute_distances(traj, [[met_ca_id, phe_cz_id], [lys_ca_id, phe_cz_id]])
    if save_to_disk is not None: np.save(save_to_disk, distances)
    return distances


def assign_dfg_spatial(dbdist, centroids=None, save_to_disk=None):
    """
    Assign Abl conformations to a one of the 3 spatial clusters based on the distance to the cluster's centroid.
    Cluster indices for DFG-in:  {0 : DFG-in,
                                  1 : DFG-inter,
                                  2 : DFG-out}
    """
    if centroids is None:
        centroids = np.load('/home/rzhu/Desktop/projects/kinase_analysis/data_egfr/cluster_centers/dfg_spatial_centroids.npy', allow_pickle=True)
    distances = cdist(dbdist, centroids)
    assignments = np.argmin(distances, axis=1)
    return assignments


def dbdihed_featuriser(traj, save_to_disk=None) -> np.ndarray:
    ########## Dunbrack DFG dihedrals ############
    # The backbone dihedrals of 
    # X-DFG (residue before conserved Asp, Thr), DFG-Asp, DFG-Phe;
    # and the side chain dihedral (χ1) of DFG-Phe
    ##############################################
    top = traj.topology

    xx_dfg_c_id = find_atomid(top, 'ILE853-C')
    x_dfg_n_id = find_atomid(top, 'THR854-N')
    x_dfg_ca_id = find_atomid(top, 'THR854-CA')
    x_dfg_c_id = find_atomid(top, 'THR854-C')
    dfg_d_n_id = find_atomid(top, 'ASP855-N')
    dfg_d_ca_id = find_atomid(top, 'ASP855-CA')
    dfg_d_c_id = find_atomid(top, 'ASP855-C')
    dfg_f_n_id = find_atomid(top, 'PHE856-N')
    dfg_f_ca_id = find_atomid(top, 'PHE856-CA')
    dfg_f_c_id = find_atomid(top, 'PHE856-C')
    dfg_g_n_id = find_atomid(top, 'GLY857-N')
    dfg_g_ca_id = find_atomid(top, 'GLY857-CA')
    dfg_g_c_id = find_atomid(top, 'GLY857-C')
    dfg_x_n_id = find_atomid(top, 'LEU858-N')
    dfg_f_cb_id = find_atomid(top, 'PHE856-CB')
    dfg_f_cg_id = find_atomid(top, 'PHE856-CG')

    dihedrals = md.compute_dihedrals(traj, 
                                     [[xx_dfg_c_id, x_dfg_n_id, x_dfg_ca_id, x_dfg_c_id], # X-DFG phi
                                      [x_dfg_n_id, x_dfg_ca_id, x_dfg_c_id, dfg_d_n_id],  # X-DFG psi
                                      [x_dfg_c_id, dfg_d_n_id, dfg_d_ca_id, dfg_d_c_id],  # DFG-D phi
                                      [dfg_d_n_id, dfg_d_ca_id, dfg_d_c_id, dfg_f_n_id],  # DFG-D psi
                                      [dfg_d_c_id, dfg_f_n_id, dfg_f_ca_id, dfg_f_c_id],  # DFG-F phi
                                      [dfg_f_n_id, dfg_f_ca_id, dfg_f_c_id, dfg_g_n_id],  # DFG-F psi
                                      [dfg_f_c_id, dfg_g_n_id, dfg_g_ca_id, dfg_g_c_id],  # DFG-G phi
                                      [dfg_g_n_id, dfg_g_ca_id, dfg_g_c_id, dfg_x_n_id],  # DFG-G psi
                                      [dfg_f_n_id, dfg_f_ca_id, dfg_f_cb_id, dfg_f_cg_id]]) # DFG-F chi1

    if save_to_disk is not None: np.save(save_to_disk, dihedrals)
    return dihedrals


def angle_distance(u1, u2):
    """
    Calculate the distance between two angles.
    """
    return 2 * (1 - np.cos(u1 - u2))


def dihedral_distance(vector1, vector2):
    """
    Calculate the total distance between two vectors of dihedral angles.
    """
    total_distance = 0
    for angle1, angle2 in zip(vector1, vector2):
        total_distance += angle_distance(angle1, angle2)
    return total_distance/len(vector1)


def assign_dfg_dihed(dbdiheds, spatial_group, centroids=None, noise_cutoff=1, save_to_disk=None):
    """
    Assign Abl conformations to a one of the seven dihedral clusters based on the distance to the cluster's centroid.
    Cluster indices for DFG-in:   {-1 : noise,
                                    0 : BLAminus,
                                    1 : BLAplus,
                                    2 : ABAminus,
                                    3 : BLBminus,
                                    4 : BLBplus,
                                    5 : BLBtrans}
    Cluster indices for DFG-inter:{-1 : noise,
                                    0 : BABtrans}
    Cluster indices for DFG-out:  {-1 : noise,
                                    0 : BBAminus}  
    """

    if centroids is None:
        centroids = np.load('/home/rzhu/Desktop/projects/kinase_analysis/data_egfr/cluster_centers/dfg_dihed_centroids.npy', allow_pickle=True).item()
    centroids = centroids[spatial_group]

    distances = []
    for dbdihed in dbdiheds:
        distances.append([dihedral_distance(dbdihed, centroid) for centroid in centroids])
    distances = np.array(distances)
    assignments = np.argmin(distances, axis=1)
    min_distances = np.min(distances, axis=1)
    assignments[min_distances > noise_cutoff] = -1

    return assignments


def compute_dfg_assignment(dists, diheds, dist_centroids=None, dihed_centroids=None, dihed_cutoff=1):
    '''
    Compute the dfg spatial group and dihedral group assignment to a group of conformations. Usually a macrostate
    '''
    
    assert dists.shape[0] == diheds.shape[0], 'Feature vectors should have the same length'
    assert dists.shape[1] == 2, 'Incorrect number of distance features, should be 2'
    assert diheds.shape[1] == 7, 'Incorrect number of dihedral features, should be 7'

    spatial_clusters = assign_dfg_spatial(dists, dist_centroids)
    spatial_counts = [sum(spatial_clusters == i) for i in range(3)]

    in_diheds, inter_diheds, out_diheds = diheds[spatial_clusters==0], diheds[spatial_clusters==1], diheds[spatial_clusters==2]
    if len(in_diheds) > 0:
        in_dihed_clusters = assign_dfg_dihed(in_diheds, 'dfg-in', dihed_centroids, dihed_cutoff)
    else:
        in_dihed_clusters = np.array([])
    if len(inter_diheds) > 0:
        inter_dihed_clusters = assign_dfg_dihed(inter_diheds, 'dfg-inter', dihed_centroids, dihed_cutoff)
    else:
        inter_dihed_clusters = np.array([])
    if len(out_diheds) > 0:
        out_dihed_clusters = assign_dfg_dihed(out_diheds, 'dfg-out', dihed_centroids, dihed_cutoff)
    else:
        out_dihed_clusters = np.array([])
        
    dihed_counts = [[sum(in_dihed_clusters == i) for i in range(-1, 6)],
                    [sum(inter_dihed_clusters == i) for i in range(-1,1)],
                    [sum(out_dihed_clusters == i) for i in range(-1,1)]]

    return spatial_counts, dihed_counts


def aloop_featuriser(traj, save_to_disk=None) -> np.ndarray:
    ########## Activation loop contacts ############
    # Pairwise distances between Cα atoms of residue i and i+3
    # Activation loop starts from ASP855 and ends at PRO877
    ################################################
    top = traj.topology

    asp855_id = np.where([str(res) == 'ASP855' for res in top.residues])[0][0]
    pro877_id = np.where([str(res) == 'PRO877' for res in top.residues])[0][0]
    aloop_traj = traj.atom_slice(top.select(f'resid {asp855_id} to {pro877_id}'))

    distances = md.compute_contacts(aloop_traj, contacts='all', scheme='ca')[0]
    if save_to_disk is not None: np.save(save_to_disk, distances)
    return distances


def ploopdihed_featuriser(traj, save_to_disk=None) -> np.ndarray:
    ########## phosphate-binding dihedral angles ############
    # Compute the phi and psi angles from LYS716 to LYS728
    ########################################################
    top = traj.topology
    ILE715_C = find_atomid(top, 'ILE715-C')
    GLY729_N = find_atomid(top, 'GLY729-N')

    ploop = traj.atom_slice(traj.topology.select(f'index {ILE715_C} to {GLY729_N}'))
    phi = md.compute_phi(ploop)[1]
    psi = md.compute_psi(ploop)[1]
    dihedrals = np.concatenate([phi, psi], axis=1)
    if save_to_disk is not None: np.save(save_to_disk, dihedrals)
    return dihedrals


def ploop_featuriser(traj, save_to_disk=None) -> np.ndarray:
    ########## phosphate-binding loop distances ############
    # d1 = dist(Cα-Leu718, Cα-Phe723) * maybe use Gly724 because it's conserved
    # d2 = dist(Cα-Gly719, Cα-Ala722)
    ########################################################
    top = traj.topology

    leu718_ca_id = find_atomid(top, 'LEU718-CA')
    phe723_ca_id = find_atomid(top, 'PHE723-CA')
    gly719_ca_id = find_atomid(top, 'GLY719-CA')
    ala722_ca_id = find_atomid(top, 'ALA722-CA')

    distances = md.compute_distances(traj, [[leu718_ca_id, phe723_ca_id], [gly719_ca_id, ala722_ca_id]])
    if save_to_disk is not None: np.save(save_to_disk, distances)
    return distances


def achelix_featuriser(traj, save_to_disk=None) -> np.ndarray:
    ########## phosphate-binding loop distances ############
    # d1 = dist(Nz-Lys745, Cd-Glu762) 
    # d2 = dist(Ce-Lys860, Cd-Glu762) 
    ########################################################
    top = traj.topology

    lys745_nz_id = find_atomid(top, 'LYS745-NZ')
    glu762_cd_id = find_atomid(top, 'GLU762-CD')
    lys860_ce_id = find_atomid(top, 'LYS860-CE')

    distances = md.compute_distances(traj, [[lys745_nz_id, glu762_cd_id], [lys860_ce_id, glu762_cd_id]])
    if save_to_disk is not None: np.save(save_to_disk, distances)
    return distances


def rspine_featuriser(traj, save_to_disk=None) -> np.ndarray:
    ########## phosphate-binding loop distances ############
    # d1 = dist(Cg-Leu777, Sd-Met766) 
    # d2 = dist(Cg-Phe856, Sd-Met766) 
    ########################################################
    top = traj.topology

    leu777_cg_id = find_atomid(top, 'LEU777-CG')
    met766_sd_id = find_atomid(top, 'MET766-SD')
    phe856_cg_id = find_atomid(top, 'PHE856-CG')

    distances = md.compute_distances(traj, [[leu777_cg_id, met766_sd_id], [phe856_cg_id, met766_sd_id]])
    if save_to_disk is not None: np.save(save_to_disk, distances)
    return distances


def aloopdssp_featuriser(traj, save_to_disk=None) -> np.ndarray:
    ########## Activation loop secondary structure ############
    # Secondary structure assignment of activation loop
    ###########################################################
    top = traj.topology

    asp855_id = np.where([str(res) == 'ASP855' for res in top.residues])[0][0]
    pro877_id = np.where([str(res) == 'PRO877' for res in top.residues])[0][0]
    aloop_traj = traj.atom_slice(top.select(f'resid {asp855_id} to {pro877_id}'))   
    
    dssp = md.compute_dssp(aloop_traj)
    #fraction_of_helix = np.mean(dssp == 'H', axis=1)

    if save_to_disk is not None: np.save(save_to_disk, dssp)
    return dssp


def achelixdssp_featuriser(traj, save_to_disk=None) -> np.ndarray:
    ########## alpha-C helix secondary structure ################
    # Secondary structure assignment of alpha-C helix
    # Alpha-C helix is defined as 
    # LYS757 to SER768
    #############################################################
    top = traj.topology

    LYS757_id = np.where([str(res) == 'LYS757' for res in top.residues])[0][0]
    SER768_id = np.where([str(res) == 'SER768' for res in top.residues])[0][0]   
    achelix_traj = traj.atom_slice(top.select(f'resid {LYS757_id} to {SER768_id}'))   
    
    dssp = md.compute_dssp(achelix_traj)
    #fraction_of_helix = np.mean(dssp == 'H', axis=1)

    if save_to_disk is not None: np.save(save_to_disk, dssp)
    return dssp


def achelixdihed_featuriser(traj, save_to_disk=None) -> np.ndarray:
    ########## alpha-C helix secondary structure ################
    # Dihedral angles of alpha-C helix
    # Alpha-C helix is defined as 
    # LYS757 to SER768
    #############################################################
    top = traj.topology
    LYS757_C = find_atomid(top, 'LYS757-C')
    SER768_N = find_atomid(top, 'SER768-N')

    achelix = traj.atom_slice(traj.topology.select(f'index {LYS757_C} to {SER768_N}'))
    phi = md.compute_phi(achelix)[1]
    psi = md.compute_psi(achelix)[1]
    dihedrals = np.concatenate([phi, psi], axis=1)

    if save_to_disk is not None: np.save(save_to_disk, dihedrals)
    return dihedrals


def featurise(featurisers: List, traj: md.Trajectory) -> np.ndarray:
    assert type(traj) == md.Trajectory
    
    ftraj_list = []
    for featuriser in featurisers:
        ftraj = featuriser(traj)
        if ftraj.dim == 1: 
            ftraj = ftraj[:, np.newaxis]
        ftraj_list.append(ftraj)

    return np.concatenate(ftraj_list, axis=1)