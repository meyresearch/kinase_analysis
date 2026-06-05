"""Featurisation and feature-selection helpers for ensemble comparison."""

import mdtraj as md
import numpy as np
import pandas as pd
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
    features : ndarray
        Concatenated feature array stacked over all frames of all trajectories
    feature_info : dict
        Dictionary mapping feature names to (start_idx, end_idx) tuples
    atom_indices_info : dict
        Dictionary mapping feature names to lists of atom indices involved
    """
    features = []
    feature_info = {}
    atom_indices_info = {}

    for traj_idx, traj in enumerate(trajectories):
        # Each block is (name, values, atom_indices); blocks are concatenated
        # column-wise in this order to form the per-frame feature vector.
        blocks = []

        # Alpha carbon pairwise distances
        ca_indices = traj.top.select('name CA')
        ca_pairs = [(ca_indices[a], ca_indices[b])
                    for a in range(len(ca_indices))
                    for b in range(a + 1, len(ca_indices))]
        blocks.append(('ca_distances', md.compute_distances(traj, ca_pairs), ca_pairs))

        # Backbone dihedrals (phi, psi) as sin/cos components
        phi_indices, phi_angles = md.compute_phi(traj)
        psi_indices, psi_angles = md.compute_psi(traj)
        blocks.extend([
            ('phi_sin', np.sin(phi_angles), phi_indices),
            ('phi_cos', np.cos(phi_angles), phi_indices),
            ('psi_sin', np.sin(psi_angles), psi_indices),
            ('psi_cos', np.cos(psi_angles), psi_indices),
        ])

        # Sidechain chi dihedrals (chi1-chi4) as sin/cos components
        chi_computers = [md.compute_chi1, md.compute_chi2, md.compute_chi3, md.compute_chi4]
        for chi_num, compute_chi in enumerate(chi_computers, start=1):
            chi_indices, chi_angles = compute_chi(traj)
            blocks.append((f'chi{chi_num}_sin', np.sin(chi_angles), chi_indices))
            blocks.append((f'chi{chi_num}_cos', np.cos(chi_angles), chi_indices))

        # Record the column layout once, using the first trajectory.
        if traj_idx == 0:
            current_dim = 0
            for name, values, atom_indices in blocks:
                feature_info[name] = (current_dim, current_dim + values.shape[1])
                atom_indices_info[name] = atom_indices
                current_dim += values.shape[1]

        features.append(np.concatenate([values for _, values, _ in blocks], axis=1))

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
    X = np.vstack([s1_features, s2_features])

    # Step 1: drop near-constant features
    vt = VarianceThreshold(threshold=var_threshold)
    X_var = vt.fit_transform(X)
    kept_idx_var = np.where(vt.get_support())[0]

    # Step 2: drop one of every pair of highly correlated features
    corr = pd.DataFrame(X_var).corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > corr_threshold)]
    keep_cols = [col for col in corr.columns if col not in to_drop]

    # Map surviving columns back to indices in the original input
    selected_idx = kept_idx_var[keep_cols]

    return selected_idx
