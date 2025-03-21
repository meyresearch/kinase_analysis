# This file hard codes the atom indices that are used in feature calculations. 
# Should we automate this for different kinases in the future? 

import numpy as np

def find_atomid(top, atom_name) -> int:
    atomid = np.where([str(atom) == atom_name for atom in top.atoms])[0]
    assert len(atomid) == 1
    return atomid[0]


def get_feature_indices(top, protein, feature):
    indices = dict()

    if protein == 'abl':
        ########## DFG spatial distances atom ids ############
        if feature == 'db_dist':
            indices['aC_M-ca']  = find_atomid(top, 'MET290-CA')
            indices['dfg_F-cz'] = find_atomid(top, 'PHE382-CZ')
            indices['b3_K-ca']  = find_atomid(top, 'LYS271-CA')
        ########## DFG dihedral atom ids ############
        if feature == 'db_dihed':
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
        if feature == 'aloop':
            indices['aloop_start'] = np.where([str(res) == 'ASP381' for res in top.residues])[0][0]
            indices['aloop_end']   = np.where([str(res) == 'ILE403' for res in top.residues])[0][0]
        ########## aC helix distances atom ids ############
        if feature == 'aChelix_dist':
            indices['b3_K-nz']      = find_atomid(top, 'LYS271-NZ')
            indices['aC_E-cd']      = find_atomid(top, 'GLU286-CD')
            indices['aloop_R/K-cz'] = find_atomid(top, 'ARG386-CZ')
        ########## aC helix start/end residue ids ############
        if feature == 'aChelix':
            indices['aC_start']    = np.where([str(res) == 'GLU281' for res in top.residues])[0][0]
            indices['aC_end']      = np.where([str(res) == 'GLU292' for res in top.residues])[0][0]
        
        ########## Features from the mechanism paper to track DFG-flip ############ 
        if feature == 'inHbond_dist':
            # A H-bond interaction between dfg-D and p-loop-Tyr observed in dfg-in conformations 
            indices['dfg_D-cg'] = find_atomid(top, 'ASP381-CG')
            indices['ploop_Hdonor'] = find_atomid(top, 'TYR253-OH')
        if feature == 'interHbond1_dist':
            # A brief H-bond interaction between dfg-D and b5-strand-Thr observed in the dfg-in to dfg-inter transition 
            indices['dfg_D-cg'] = find_atomid(top, 'ASP381-CG')
            indices['b5_T-og1'] = find_atomid(top, 'THR315-OG1')
        if feature == 'interHbond2_dist':
            # A stable H-bond interaction between the Asp381 and Val299 backbone oxygen 
            # in the dfg-inter conformations
            # Asp not protonated -- will the H-bond still there? 
            indices['dfg_D-cg'] = find_atomid(top, 'ASP381-CG')
            indices['b4_backbone'] = find_atomid(top, 'VAL299-O')
        if feature == 'interpipi_dist':
            # A brief pi-pi stacking between dfg-f and hrd-h formed during dfg-inter -> out transition
            indices['dfg_F-cg'] = find_atomid(top, 'PHE382-CG')
            indices['hrd_H-ne2'] = find_atomid(top, 'HIS361-NE2')
        if feature == 'outpipi_dist':
            # Pi-Pi stacking between dfg-f and p-loop aromatic in DFG-out conformations 
            indices['dfg_F-cg'] = find_atomid(top, 'PHE382-CG')
            indices['ploop_ring-cg'] = find_atomid(top, 'TYR253-CG')
        if feature == 'pathway_angle':
            # The angle formed by dfg-F CZ, dfg-D CG, and a close lysine backbone O
            # This angle should either increase or decrease during the flipping
            # with peak differences at intermediate DFG configurations
            indices['dfg_F-cg'] = find_atomid(top, 'PHE382-CG')
            indices['dfg_D-cg'] = find_atomid(top, 'ASP381-CG')
            indices['(D-3)_K-O'] = find_atomid(top, 'LYS378-O')
    elif protein == 'egfr':
        ########## DFG spatial distances atom ids ############
        if feature == 'db_dist':
            indices['aC_M-ca']   = find_atomid(top, 'MET766-CA')
            indices['dfg_F-cz']  = find_atomid(top, 'PHE856-CZ')
            indices['b3_K-ca']   = find_atomid(top, 'LYS745-CA')
        ########## DFG dihedral atom ids ############
        if feature == 'db_dihed':
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
        if feature == 'aloop':
            indices['aloop_start'] = np.where([str(res) == 'ASP855' for res in top.residues])[0][0]
            indices['aloop_end']   = np.where([str(res) == 'PRO877' for res in top.residues])[0][0]
        ########## aC helix distances atom ids ############
        if feature == 'aChelix_dist':
            indices['b3_K-nz']        = find_atomid(top, 'LYS745-NZ')
            indices['aC_E-cd']        = find_atomid(top, 'GLU762-CD')
            indices['aloop_R/K-cz']   = find_atomid(top, 'LYS860-CE')
        ########## aC helix start/end residue ids ############
        if feature == 'aChelix':
            indices['aC_start']    = np.where([str(res) == 'LYS757' for res in top.residues])[0][0]
            indices['aC_end']      = np.where([str(res) == 'SER768' for res in top.residues])[0][0]

        ########## Features from the mechanism paper to track DFG-flip ############ 
        if feature == 'inHbond_dist':
            # The Tyr found in Abl1 is replaced by a Phe in EGFR 
            # Consequently there's shouldn't be any H-bond
            # We still measure the distance between Asp and Phe as a comparison 
            indices['dfg_D-cg'] = find_atomid(top, 'ASP855-CG')
            indices['ploop_Hdonor'] = find_atomid(top, 'PHE723-CZ')
        if feature == 'interHbond1_dist':
            # A brief H-bond interaction between dfg-D and b5-strand-Thr observed in the dfg-in to dfg-inter transition 
            indices['dfg_D-cg'] = find_atomid(top, 'ASP855-CG')
            indices['b5_T-og1'] = find_atomid(top, 'THR790-OG1')
        if feature == 'interHbond2_dist':
            # A stable H-bond interaction between the Asp and Cys(?) backbone oxygen 
            # in the dfg-inter conformations
            # Asp not protonated -- will the H-bond still there? 
            indices['dfg_D-cg'] = find_atomid(top, 'ASP855-CG')
            indices['b4_backbone'] = find_atomid(top, 'CYS775-O')
        if feature == 'interpipi_dist':
            # A brief pi-pi stacking between dfg-f and hrd-h 
            indices['dfg_F-cg'] = find_atomid(top, 'PHE856-CG')
            indices['hrd_H-ne2'] = find_atomid(top, 'HIS835-NE2')
        if feature == 'outpipi_dist':
            # Pi-Pi stacking between dfg-f and p-loop aromatic in DFG-out conformations 
            indices['dfg_F-cg'] = find_atomid(top, 'PHE856-CG')
            indices['ploop_ring-cg'] = find_atomid(top, 'PHE723-CG')
        if feature == 'pathway_angle':
            # The angle formed by dfg-F CZ, dfg-D CG, and a close lysine backbone O
            # This angle should either increase or decrease during the flipping
            # with peak differences at intermediate DFG configurations
            indices['dfg_F-cg'] = find_atomid(top, 'PHE856-CG')
            indices['dfg_D-cg'] = find_atomid(top, 'ASP855-CG')
            indices['(D-3)_K-O'] = find_atomid(top, 'LYS852-O')
    elif 'met' in protein :
        index_offset = 0 # default to neither but define variable for clarity
        if 'met-af2' in protein:
            index_offset = 1067
        elif 'met-pdb' in protein:
            index_offset = 1066
    ########## DFG spatial distances atom ids ############
        if feature == 'db_dist':
            indices['aC_M-ca']  = find_atomid(top, f'MET{1131-index_offset}-CA')
            indices['dfg_F-cz'] = find_atomid(top, f'PHE{1223-index_offset}-CZ')
            indices['b3_K-ca']  = find_atomid(top, f'LYS{1110-index_offset}-CA')
        ########## DFG dihedral atom ids ############
        if feature == 'db_dihed':
            indices['XX_dfg-c']   = find_atomid(top, f'VAL{1220-index_offset}-C')
            indices['X_dfg-n']    = find_atomid(top, f'ALA{1221-index_offset}-N')
            indices['X_dfg-ca']   = find_atomid(top, f'ALA{1221-index_offset}-CA')
            indices['X_dfg-c']    = find_atomid(top, f'ALA{1221-index_offset}-C')
            indices['dfg_D-n']    = find_atomid(top, f'ASP{1222-index_offset}-N')
            indices['dfg_D-ca']   = find_atomid(top, f'ASP{1222-index_offset}-CA')
            indices['dfg_D-c']    = find_atomid(top, f'ASP{1222-index_offset}-C')
            indices['dfg_F-n']    = find_atomid(top, f'PHE{1223-index_offset}-N')
            indices['dfg_F-ca']   = find_atomid(top, f'PHE{1223-index_offset}-CA')
            indices['dfg_F-c']    = find_atomid(top, f'PHE{1223-index_offset}-C')
            indices['dfg_G-n']    = find_atomid(top, f'GLY{1224-index_offset}-N')
            indices['dfg_G-ca']   = find_atomid(top, f'GLY{1224-index_offset}-CA')
            indices['dfg_G-c']    = find_atomid(top, f'GLY{1224-index_offset}-C')
            indices['dfg_X-n']    = find_atomid(top, f'LEU{1225-index_offset}-N')
            indices['dfg_F-cb']   = find_atomid(top, f'PHE{1223-index_offset}-CB')
            indices['dfg_F-cg']   = find_atomid(top, f'PHE{1223-index_offset}-CG')
        ########## Activation loop start/end residue ids ############
        if feature == 'aloop':
            indices['aloop_start'] = np.where([str(res) == f'ASP{1222-index_offset}' for res in top.residues])[0][0]
            indices['aloop_end']   = np.where([str(res) == f'VAL{1247-index_offset}' for res in top.residues])[0][0]
        ########## aC helix distances atom ids ############
        if feature == 'aChelix_dist':
            indices['b3_K-nz']      = find_atomid(top, f'LYS{1110-index_offset}-NZ')
            indices['aC_E-cd']      = find_atomid(top, f'GLU{1127-index_offset}-CD')
            indices['aloop_R/K-cz'] = find_atomid(top, f'ARG{1227-index_offset}-CZ')
        ########## aC helix start/end residue ids ############
        if feature == 'aChelix':
            indices['aC_start']    = np.where([str(res) == f'SER{1122-index_offset}' for res in top.residues])[0][0]
            indices['aC_end']      = np.where([str(res) == f'ASP{1133-index_offset}' for res in top.residues])[0][0]
        
        ########## Features from the mechanism paper to track DFG-flip ############ 
        if feature == 'inHbond_dist':
            # A H-bond interaction between dfg-D and p-loop-Tyr observed in dfg-in conformations
            # The Tyr found in Abl1 is replaced by a Phe in MET 
            # Consequently there's shouldn't be any H-bond
            # We still measure the distance between Asp and Phe as a comparison  
            indices['dfg_D-cg'] = find_atomid(top, f'ASP{1222-index_offset}-CG')
            indices['ploop_Hdonor'] = find_atomid(top, f'PHE{1089-index_offset}-CZ')
        if feature == 'interHbond1_dist':
            # A brief H-bond interaction between dfg-D and b5-strand-Thr observed in the dfg-in to dfg-inter transition
            # The Thr found in Abl1 is replaced by a Leu in MET 
            indices['dfg_D-cg'] = find_atomid(top, f'ASP{1222-index_offset}-CG')
            # indices['b5_T-og1'] = find_atomid(top, 'THR315-OG1')
        if feature == 'interHbond2_dist':
            # A stable H-bond interaction between the Asp381 and Val299 backbone oxygen 
            # in the dfg-inter conformations
            # Asp not protonated -- will the H-bond still there? 
            indices['dfg_D-cg'] = find_atomid(top, f'ASP{1222-index_offset}-CG')
            indices['b4_backbone'] = find_atomid(top, f'LEU{1140-index_offset}-O')
        if feature == 'interpipi_dist':
            # A brief pi-pi stacking between dfg-f and hrd-h formed during dfg-inter -> out transition
            indices['dfg_F-cg'] = find_atomid(top, f'PHE{1223-index_offset}-CG')
            indices['hrd_H-ne2'] = find_atomid(top, f'HIS{1202-index_offset}-NE2')
        if feature == 'outpipi_dist':
            # Pi-Pi stacking between dfg-f and p-loop aromatic in DFG-out conformations 
            indices['dfg_F-cg'] = find_atomid(top, f'PHE{1223-index_offset}-CG')
            indices['ploop_ring-cg'] = find_atomid(top, f'PHE{1089-index_offset}-CG')
        if feature == 'pathway_angle':
            # The angle formed by dfg-F CZ, dfg-D CG, and a close lysine backbone O
            # This angle should either increase or decrease during the flipping
            # with peak differences at intermediate DFG configurations
            indices['dfg_F-cg'] = find_atomid(top, f'PHE{1223-index_offset}-CG')
            indices['dfg_D-cg'] = find_atomid(top, f'ASP{1222-index_offset}-CG')
            indices['(D-3)_K-O'] = find_atomid(top, f'LYS{1219-index_offset}-O')
    else:
        raise ValueError('Protein name cannot be recognised.')

    return indices