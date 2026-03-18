import numpy as np
from scipy.spatial.distance import cdist
from typing import *
import pickle
import hdbscan
from funcs_featurise import dbdist_featuriser, dbdihed_featuriser


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


def assign_dfg_spatial_hdbscan(dbdist, hdbscan_model, mapping=None, confidence_threshold=0.01):
    """
    Assign conformations to one of the 3 spatial clusters using HDBSCAN approximate prediction.

    Original cluster indices:
        0 : DFG-in
        1 : DFG-inter
        2 : DFG-out

    Parameters
    ----------
    dbdist : array-like, shape (n_samples, n_features)
        Input features.
    hdbscan_model : str or fitted HDBSCAN object
        Path to pickled HDBSCAN model, or the fitted model itself.
    mapping : dict, optional
        Mapping from original labels to desired labels, e.g.
        {-1: -1, 0: 1, 1: 2, 2: 0}
    confidence_threshold : float, optional
        Points with prediction strength below this threshold are assigned to -1.

    Returns
    -------
    assignments : np.ndarray
        Final assigned labels.
    """
    if isinstance(hdbscan_model, str):
        with open(hdbscan_model, "rb") as f:
            model = pickle.load(f)
    else:
        model = hdbscan_model

    labels, strength = hdbscan.approximate_predict(model, dbdist)

    assignments = labels.copy()
    assignments[strength < confidence_threshold] = -1

    if mapping is not None:
        assignments = np.array([mapping[x] for x in assignments])

    return assignments


# def assign_dfg_spatial(dbdist, centroids=None):
#     """
#     Assign conformations to a one of the 3 spatial clusters based on the distance to the cluster's centroid.
#     Cluster indices for DFG spatial groups: 0 : DFG-in,
#                                             1 : DFG-inter,
#                                             2 : DFG-out
#     """
#     if centroids is None:
#         centroids = np.load('/home/rzhu/Desktop/projects/kinase_analysis/data/abl/clustering/dfg_spatial_centroids.npy', allow_pickle=True)
#     distances = cdist(dbdist, centroids)
#     assignments = np.argmin(distances, axis=1)
#     return assignments


def assign_ramachandran_codes(dbdihed_list):
    # Get 4 letter codes for X, D, F Ramachandran assignment

    centroids = {
    "A": np.array([-np.pi/3, -np.pi/4]),   # (-60°, -45°)
    "B": np.array([-2*np.pi/3,  2*np.pi/3]),  # (-120°, 120°)
    "L": np.array([ np.pi/3,  np.pi/6])    # (60°, 30°)
    }
    threshold = np.pi

    def angular_diff(a, b):
        return np.arctan2(np.sin(a - b), np.cos(a - b))
    
    def ramachandran_distance(phi, psi, centroid):
        dphi  = angular_diff(phi, centroid[0])
        dpsi  = angular_diff(psi, centroid[1])
        return np.sqrt(dphi**2 + dpsi**2)
    
    def distances_to_centroids(phi, psi, centroids):
        return {
            name: ramachandran_distance(phi, psi, center)
            for name, center in centroids.items()
        }
    
    def get_codes(distance_dict, threshold, default = 'X'):
        labels = list(distance_dict.keys())
        distance_arrays = [np.asarray(distance_dict[label]) for label in labels]

        stacked = np.stack(distance_arrays, axis=0)
        min_indices = stacked.argmin(axis=0)
        min_values = stacked.min(axis=0)

        codes = []
        for idx, value in zip(min_indices.ravel(), min_values.ravel()):
            codes.append(labels[idx] if value < threshold else default)

        return codes

    x_distances, d_distances, f_distances, f_chi_distances, codes = [], [], [], [], []
    for dihedrals in dbdihed_list:
        x_phi, x_psi = dihedrals[:, 0], dihedrals[:, 1]
        d_phi, d_psi = dihedrals[:, 2], dihedrals[:, 3]
        f_phi, f_psi = dihedrals[:, 4], dihedrals[:, 5]
        f_chi = dihedrals[:, -1]

        x_distance = distances_to_centroids(x_phi, x_psi, centroids)
        d_distance = distances_to_centroids(d_phi, d_psi, centroids)
        f_distance = distances_to_centroids(f_phi, f_psi, centroids)
        f_chi_distance = {
            name: np.abs(angular_diff(f_chi, center))
            for name, center in {'plus':np.pi/3, 'minus':-np.pi/3, 'trans':np.pi}.items()
        }
        
        x_distances.append(x_distance)
        d_distances.append(d_distance)
        f_distances.append(f_distance)
        f_chi_distances.append(f_chi_distance)

        x_code = get_codes(x_distance, threshold)
        d_code = get_codes(d_distance, threshold)
        f_code = get_codes(f_distance, threshold)
        f_chi_code = get_codes(f_chi_distance, threshold)
        codes.append([''.join(group) for group in zip(x_code, d_code, f_code, f_chi_code)])

    return x_distances, d_distances, f_distances, f_chi_distances, codes


def assign_dfg_dihed(dbdihed, dist_group, centroids, noise_cutoff=1):
    """
    Assign a SINGLE conformation to a one of the seven dihedral clusters based on the distance to the cluster's centroid.
    Cluster indices:   -1 : noise,
                        --- DFG-in ---
                        0 : BLAminus,
                        1 : BLAplus,
                        2 : ABAminus,
                        3 : BLBminus,
                        4 : BLBplus,
                        5 : BLBtrans
                        --- DFG-inter ---
                        6 : BABtrans
                        --- DFG-out ---
                        7 : BBAminus}
    """

    centroids = np.load(centroids, allow_pickle=True).item()
    centroids = centroids[['dfg-in', 'dfg-inter', 'dfg-out'][dist_group]]
    distance = np.array([dihedral_distance(dbdihed, centroid) for centroid in centroids])

    if np.min(distance) > noise_cutoff or dist_group == -1:
        return -1
    elif dist_group == 0:  # DFG-in group indexing from 0
        return np.argmin(distance)
    elif dist_group == 1:  # DFG-inter group indexing from 6
        return np.argmin(distance) + 6
    elif dist_group == 2:  # DFG-out group indexing from 7
        return np.argmin(distance) + 7


def dfg_featuriser(traj, protein, 
                   hdbscan_model, dihed_centroids, mapping=None,
                   save_to_disk=None, 
                   confidence_threshold=0.01, dihed_cutoff=1, 
                   spatial_name='dist_group', dihed_name='dihed_group') -> list[np.ndarray]:
    '''
    Compute the dfg dihedral group assignment to structures based on Dunbrack distances and dihedrals.

    Parameters
    ----------
    traj : mdtraj.Trajectory
        The trajectory to be featurised
    protein : str
        The protein name that is used to select atomic indices to compute distances and dihedrals
    hdbscan_model : Path
        The path to the HDBSCAN model
    dihed_centroids : Path
        The path to the dihedral centroids
    save_to_disk : Path
        The directory to save the spatial and dihedral assignments
    confidence_threshold : float
        The confidence threshold for the HDBSCAN clustering
    dihed_cutoff : float
        The cutoff distance for the dihedral cluster assignment
    spatial_name : str
        The name of the dfg-in, dfg-inter, dfg-out spatial assignment directory
    dihed_name : str
        The name of the dihedral assignment directory
    
    Returns
    -------
    spatial_assignments : np.ndarray
        The spatial assignments of each snapshot
    dihed_assignments : np.ndarray
        The dihedral assignments of each snapshot
    '''

    dists = dbdist_featuriser(traj, protein)
    diheds = dbdihed_featuriser(traj, protein)[:, [0,1,2,3,4,5,8]]

    assert dists.shape[0] == diheds.shape[0], 'Feature vectors should have the same length'
    assert dists.shape[1] == 2, 'Incorrect number of distance features, should be 2'
    assert diheds.shape[1] == 7, 'Incorrect number of dihedral features, should be 7'

    spatial_assignments = assign_dfg_spatial_hdbscan(dists, hdbscan_model, mapping, confidence_threshold)
    dihed_assignments = np.array(list(map(lambda ds: assign_dfg_dihed(ds[0], ds[1], dihed_centroids, dihed_cutoff), zip(diheds, spatial_assignments))))
    if save_to_disk is not None: 
        dir = save_to_disk.parent
        stem = save_to_disk.stem.split('_')[0]
        np.save(dir/f'{stem}_{spatial_name}.npy', spatial_assignments)
        np.save(dir/f'{stem}_{dihed_name}.npy', dihed_assignments)

    return spatial_assignments, dihed_assignments


def dunbrack_count(spatial_assignments, dihedral_assignments):
    '''
    Count the number of snapshots in each spatial and dihedral cluster.

    Parameters
    ----------
    spatial_assignments : np.ndarray
        The spatial assignments of each snapshot
    dihedral_assignments : np.ndarray
        The dihedral assignments of each snapshot
    '''
    
    spatial_counts = np.array([np.sum(spatial_assignments == i) for i in range(-1, 3)])
    in_dihed_counts = [np.sum(dihedral_assignments == i) for i in range(0,6)]
    inter_dihed_counts = [np.sum(dihedral_assignments == i) for i in range(6,7)]
    out_dihed_counts = [np.sum(dihedral_assignments == i) for i in range(7,8)]
    dihed_counts = [[spatial_counts[0]], 
                    [spatial_counts[1] - np.sum(in_dihed_counts)] + in_dihed_counts, 
                    [spatial_counts[2] - np.sum(inter_dihed_counts)] + inter_dihed_counts, 
                    [spatial_counts[3] - np.sum(out_dihed_counts)] + out_dihed_counts] 
    return spatial_counts, dihed_counts