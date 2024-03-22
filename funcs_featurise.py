import mdtraj as md
import numpy as np
from typing import *


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
    
    '''
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
    '''

    if save_to_disk is not None: np.save(save_to_disk, dihedrals)
    return dihedrals


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


def featurise(featurisers: List, traj: md.Trajectory) -> np.ndarray:
    assert type(traj) == md.Trajectory
    
    ftraj_list = []
    for featuriser in featurisers:
        ftraj = featuriser(traj)
        if ftraj.dim == 1: 
            ftraj = ftraj[:, np.newaxis]
        ftraj_list.append(ftraj)

    return np.concatenate(ftraj_list, axis=1)
