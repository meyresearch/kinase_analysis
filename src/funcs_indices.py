# This file hard codes the atom indices that are used in feature calculations. 
# Can we automate this for different kinases in the future? (potentially making use of the sequence alignment)

import numpy as np

def find_atomid(top, atom_name) -> int:
    atomid = np.where([str(atom) == atom_name for atom in top.atoms])[0]
    assert len(atomid) == 1
    return atomid[0]


def protein_feature_indies(top, protein):
    indices = dict()

    if protein == 'abl':
        ########## DFG spatial distances atom ids ############
        indices['aC_M-ca']  = find_atomid(top, 'MET290-CA')
        indices['dfg_F-cz']   = find_atomid(top, 'PHE382-CZ')
        indices['b3_K-ca']  = find_atomid(top, 'LYS271-CA')
        ########## DFG dihedral atom ids ############
        indices['XX_dfg-c']   = find_atomid(top, 'VAL379-C')
        indices['X_dfg-n']    = find_atomid(top, 'ALA380-N')
        indices['X_dfg-ca']   = find_atomid(top, 'ALA380-CA')
        indices['X_dfg-c']    = find_atomid(top, 'ALA380-C')
        indices['dfg_D-n']    = find_atomid(top, 'ASP381-N')
        indices['dfg_D-ca']   = find_atomid(top, 'ASP381-CA')
        indices['dfg_D-c']    = find_atomid(top, 'ASP381-C')
        indices['dfg_F-n']    = find_atomid(top, 'PHE382-N')
        indices['dfg_F-ca']   = find_atomid(top, 'PHE382-CA')
        indices['dfg_F-c']    = find_atomid(top, 'PHE382-C')
        indices['dfg_G-n']    = find_atomid(top, 'GLY383-N')
        indices['dfg_G-ca']   = find_atomid(top, 'GLY383-CA')
        indices['dfg_G-c']    = find_atomid(top, 'GLY383-C')
        indices['dfg_X-n']    = find_atomid(top, 'LEU384-N')
        indices['dfg_F-cb']   = find_atomid(top, 'PHE382-CB')
        indices['dfg_F-cg']   = find_atomid(top, 'PHE382-CG')
        ########## Activation loop start/end residue ids ############
        indices['aloop_start'] = np.where([str(res) == 'ASP381' for res in top.residues])[0][0]
        indices['aloop_end']   = np.where([str(res) == 'ILE403' for res in top.residues])[0][0]
        ########## aC helix distances atom ids ############
        indices['b3_K-nz']   = find_atomid(top, 'LYS271-NZ')
        indices['aC_E-cd']        = find_atomid(top, 'GLU286-CD')
        indices['aloop_R/K-cz']   = find_atomid(top, 'ARG386-CZ')
        ########## aC helix start/end residue ids ############
        indices['aC_start']    = np.where([str(res) == 'GLU281' for res in top.residues])[0][0]
        indices['aC_end']      = np.where([str(res) == 'GLU292' for res in top.residues])[0][0]

    elif protein == 'egfr':
        ########## DFG spatial distances atom ids ############
        indices['aC_M-ca']   = find_atomid(top, 'MET766-CA')
        indices['dfg_F-cz']   = find_atomid(top, 'PHE856-CZ')
        indices['b3_K-ca']   = find_atomid(top, 'LYS745-CA')
        ########## DFG dihedral atom ids ############
        indices['XX_dfg-c']   = find_atomid(top, 'ILE853-C')
        indices['X_dfg-n']    = find_atomid(top, 'THR854-N')
        indices['X_dfg-ca']   = find_atomid(top, 'THR854-CA')
        indices['X_dfg-c']    = find_atomid(top, 'THR854-C')
        indices['dfg_D-n']    = find_atomid(top, 'ASP855-N')
        indices['dfg_D-ca']   = find_atomid(top, 'ASP855-CA')
        indices['dfg_D-c']    = find_atomid(top, 'ASP855-C')
        indices['dfg_F-n']    = find_atomid(top, 'PHE856-N')
        indices['dfg_F-ca']   = find_atomid(top, 'PHE856-CA')
        indices['dfg_F-c']    = find_atomid(top, 'PHE856-C')
        indices['dfg_G-n']    = find_atomid(top, 'GLY857-N')
        indices['dfg_G-ca']   = find_atomid(top, 'GLY857-CA')
        indices['dfg_G-c']    = find_atomid(top, 'GLY857-C')
        indices['dfg_X-n']    = find_atomid(top, 'LEU858-N')
        indices['dfg_F-cb']   = find_atomid(top, 'PHE856-CB')
        indices['dfg_F-cg']   = find_atomid(top, 'PHE856-CG')
        ########## Activation loop start/end residue ids ############
        indices['aloop_start'] = np.where([str(res) == 'ASP855' for res in top.residues])[0][0]
        indices['aloop_end']   = np.where([str(res) == 'PRO877' for res in top.residues])[0][0]
        ########## aC helix distances atom ids ############
        indices['b3_K-nz']        = find_atomid(top, 'LYS745-NZ')
        indices['aC_E-cd']        = find_atomid(top, 'GLU762-CD')
        indices['aloop_R/K-cz']   = find_atomid(top, 'LYS860-CE')
        ########## aC helix start/end residue ids ############
        indices['aC_start']    = np.where([str(res) == 'LYS757' for res in top.residues])[0][0]
        indices['aC_end']      = np.where([str(res) == 'SER768' for res in top.residues])[0][0]

    else:
        raise ValueError('Protein name cannot be recognised.')

    return indices