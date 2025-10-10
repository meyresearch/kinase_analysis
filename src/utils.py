import mdtraj as md
import numpy as np
import pandas as pd
import pickle
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.feature_selection import VarianceThreshold


def featurize_trajectories(trajectories):
    """
    Featurize a list of trajectories by computing:
    - Alpha carbon pairwise distances
    - Backbone dihedral angles (phi, psi) converted to sin/cos components
    - Sidechain chi angles (chi1, chi2, chi3, chi4) converted to sin/cos components
    
    Parameters
    ----------
    trajectories : list of mdtraj.Trajectory
        List of trajectories to featurize
        
    Returns
    -------
    features : list of ndarray
        List of feature arrays, one per trajectory
    feature_info : dict
        Dictionary mapping feature names to (start_idx, end_idx) tuples
    atom_indices_info : dict
        Dictionary mapping feature names to lists of atom indices involved
    """
    features = []
    feature_info = {}
    atom_indices_info = {}
    
    for i, traj in enumerate(trajectories):
        features_list = []
        current_dim = 0
        
        # Alpha carbon pairwise distances
        ca_indices = traj.top.select('name CA')
        ca_pairs = [(ca_indices[i], ca_indices[j]) for i in range(len(ca_indices)) for j in range(i+1, len(ca_indices))]
        ca_distances = md.compute_distances(traj, ca_pairs)
        features_list.append(ca_distances)
        
        if i == 0:  # Only set feature_info once
            feature_info['ca_distances'] = (current_dim, current_dim + ca_distances.shape[1])
            atom_indices_info['ca_distances'] = ca_pairs
            current_dim += ca_distances.shape[1]
        
        # Backbone dihedral angles (phi, psi) - convert to sin/cos
        phi_indices, phi_angles = md.compute_phi(traj)
        psi_indices, psi_angles = md.compute_psi(traj)
        
        # Convert to sin/cos components
        phi_sin = np.sin(phi_angles)
        phi_cos = np.cos(phi_angles)
        psi_sin = np.sin(psi_angles)
        psi_cos = np.cos(psi_angles)
        
        features_list.extend([phi_sin, phi_cos, psi_sin, psi_cos])
        
        if i == 0:
            feature_info['phi_sin'] = (current_dim, current_dim + phi_sin.shape[1])
            atom_indices_info['phi_sin'] = phi_indices
            current_dim += phi_sin.shape[1]
            feature_info['phi_cos'] = (current_dim, current_dim + phi_cos.shape[1])
            atom_indices_info['phi_cos'] = phi_indices
            current_dim += phi_cos.shape[1]
            feature_info['psi_sin'] = (current_dim, current_dim + psi_sin.shape[1])
            atom_indices_info['psi_sin'] = psi_indices
            current_dim += psi_sin.shape[1]
            feature_info['psi_cos'] = (current_dim, current_dim + psi_cos.shape[1])
            atom_indices_info['psi_cos'] = psi_indices
            current_dim += psi_cos.shape[1]
        
        # Sidechain chi angles - convert to sin/cos
        chi1_indices, chi1_angles = md.compute_chi1(traj)
        chi2_indices, chi2_angles = md.compute_chi2(traj)
        chi3_indices, chi3_angles = md.compute_chi3(traj)
        chi4_indices, chi4_angles = md.compute_chi4(traj)
        
        # Convert to sin/cos components
        chi1_sin, chi1_cos = np.sin(chi1_angles), np.cos(chi1_angles)
        chi2_sin, chi2_cos = np.sin(chi2_angles), np.cos(chi2_angles)
        chi3_sin, chi3_cos = np.sin(chi3_angles), np.cos(chi3_angles)
        chi4_sin, chi4_cos = np.sin(chi4_angles), np.cos(chi4_angles)
        
        features_list.extend([chi1_sin, chi1_cos, chi2_sin, chi2_cos, 
                            chi3_sin, chi3_cos, chi4_sin, chi4_cos])
        
        if i == 0:
            feature_info['chi1_sin'] = (current_dim, current_dim + chi1_sin.shape[1])
            atom_indices_info['chi1_sin'] = chi1_indices
            current_dim += chi1_sin.shape[1]
            feature_info['chi1_cos'] = (current_dim, current_dim + chi1_cos.shape[1])
            atom_indices_info['chi1_cos'] = chi1_indices
            current_dim += chi1_cos.shape[1]
            feature_info['chi2_sin'] = (current_dim, current_dim + chi2_sin.shape[1])
            atom_indices_info['chi2_sin'] = chi2_indices
            current_dim += chi2_sin.shape[1]
            feature_info['chi2_cos'] = (current_dim, current_dim + chi2_cos.shape[1])
            atom_indices_info['chi2_cos'] = chi2_indices
            current_dim += chi2_cos.shape[1]
            feature_info['chi3_sin'] = (current_dim, current_dim + chi3_sin.shape[1])
            atom_indices_info['chi3_sin'] = chi3_indices
            current_dim += chi3_sin.shape[1]
            feature_info['chi3_cos'] = (current_dim, current_dim + chi3_cos.shape[1])
            atom_indices_info['chi3_cos'] = chi3_indices
            current_dim += chi3_cos.shape[1]
            feature_info['chi4_sin'] = (current_dim, current_dim + chi4_sin.shape[1])
            atom_indices_info['chi4_sin'] = chi4_indices
            current_dim += chi4_sin.shape[1]
            feature_info['chi4_cos'] = (current_dim, current_dim + chi4_cos.shape[1])
            atom_indices_info['chi4_cos'] = chi4_indices
        
        # Concatenate all features
        concatenated_features = np.concatenate(features_list, axis=1)
        features.append(concatenated_features)
    
    features = np.concatenate(features, axis=0)
    return features, feature_info, atom_indices_info

def get_feature_atoms(feature_dim, feature_info, atom_indices_info, reference_traj):
    """
    Get the atom information for a specific feature dimension
    
    Parameters
    ----------
    feature_dim : int
        The feature dimension index
    feature_info : dict
        Feature dimension mapping
    atom_indices_info : dict  
        Atom indices for each feature type
    reference_traj : mdtraj.Trajectory
        Reference trajectory for atom/residue names
        
    Returns
    -------
    feature_type : str
        Type of feature (e.g., 'ca_distances', 'phi_sin', 'chi1_cos')
    atoms_info : str
        Human-readable description of the atoms involved
    atom_indices : tuple or list
        The actual atom indices
    """
    
    # Find which feature type this dimension belongs to
    for feature_type, (start, end) in feature_info.items():
        if start <= feature_dim < end:
            relative_idx = feature_dim - start
            atom_indices = atom_indices_info[feature_type][relative_idx]
            
            # Create human-readable description
            if feature_type == 'ca_distances':
                atom1, atom2 = atom_indices
                res1 = reference_traj.topology.atom(atom1).residue
                res2 = reference_traj.topology.atom(atom2).residue
                atoms_info = f"CA {res1.name}{res1.resSeq} - CA {res2.name}{res2.resSeq}"
                
            elif feature_type in ['phi_sin', 'phi_cos', 'psi_sin', 'psi_cos', 
                                'chi1_sin', 'chi1_cos', 'chi2_sin', 'chi2_cos', 
                                'chi3_sin', 'chi3_cos', 'chi4_sin', 'chi4_cos']:
                atom1, atom2, atom3, atom4 = atom_indices
                
                if feature_type.startswith('phi_'):
                    res = reference_traj.topology.atom(atom2).residue
                    component = 'sin' if feature_type.endswith('_sin') else 'cos'
                    atoms_info = f"Phi {res.name}{res.resSeq} ({component})"
                elif feature_type.startswith('psi_'):
                    res = reference_traj.topology.atom(atom1).residue  
                    component = 'sin' if feature_type.endswith('_sin') else 'cos'
                    atoms_info = f"Psi {res.name}{res.resSeq} ({component})"
                else:  # chi angles
                    res = reference_traj.topology.atom(atom1).residue
                    chi_num = feature_type.split('_')[0].replace('chi', '')
                    component = 'sin' if feature_type.endswith('_sin') else 'cos'
                    atoms_info = f"Chi{chi_num} {res.name}{res.resSeq} ({component})"
                    
            return feature_type, atoms_info, atom_indices
    
    return None, None, None


def feature_cleanup_with_stats(s1_features, s2_features,
                               var_threshold=1e-3,
                               corr_threshold=0.90):
    """
    Cleanup + univariate scoring for feature arrays from two states.

    Steps:
      1. Remove low-variance features
      2. Remove highly correlated features

    Parameters
    ----------
    s1_features : ndarray, shape (n1, n_features)
    s2_features : ndarray, shape (n2, n_features)
    var_threshold : float
        Variance cutoff for dropping features
    corr_threshold : float
        Correlation cutoff for dropping redundant features

    Returns
    -------
    selected_idx : ndarray
        Indices of surviving features (relative to original input)
    """
    # Combine features and labels
    X = np.vstack([s1_features, s2_features])
    y = np.array([0]*len(s1_features) + [1]*len(s2_features))

    n_features = X.shape[1]

    # ---------------------------
    # Step 1: Variance filter
    # ---------------------------
    vt = VarianceThreshold(threshold=var_threshold)
    X_var = vt.fit_transform(X)
    mask_var = vt.get_support()
    kept_idx_var = np.where(mask_var)[0]

    # ---------------------------
    # Step 2: Correlation pruning
    # ---------------------------
    df_var = pd.DataFrame(X_var)
    corr = df_var.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    to_drop = [col for col in upper.columns if any(upper[col] > corr_threshold)]
    keep_cols = [col for col in df_var.columns if col not in to_drop]

    # surviving indices mapped back to original
    selected_idx = kept_idx_var[keep_cols]

    return selected_idx
