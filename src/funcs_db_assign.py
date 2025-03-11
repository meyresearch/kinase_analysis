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


def assign_dfg_spatial_hdbscan(dbdist, protein, confidence_threshold=0.01):
    """
    Assign conformations to a one of the 3 spatial clusters based on the distance to the cluster's centroid.
    Cluster indices for DFG spatial groups: 0 : DFG-in,
                                            1 : DFG-inter,
                                            2 : DFG-out
    """
    model = pickle.load(open(f'/home/rzhu/Desktop/projects/kinase_analysis/data/{protein}/clustering/hdbscan.pkl', 'rb'))
    
    labels, strength = hdbscan.approximate_predict(model, dbdist)
    assignments = labels.copy()
    assignments[strength < confidence_threshold] = -1
    return assignments


def assign_dfg_spatial(dbdist, centroids=None):
    """
    Assign conformations to a one of the 3 spatial clusters based on the distance to the cluster's centroid.
    Cluster indices for DFG spatial groups: 0 : DFG-in,
                                            1 : DFG-inter,
                                            2 : DFG-out
    """
    if centroids is None:
        centroids = np.load('/home/rzhu/Desktop/projects/kinase_analysis/data/abl/clustering/dfg_spatial_centroids.npy', allow_pickle=True)
    distances = cdist(dbdist, centroids)
    assignments = np.argmin(distances, axis=1)
    return assignments


def assign_dfg_dihed(dbdihed, dist_group, centroids=None, noise_cutoff=1):
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

    if centroids is None:
        centroids = np.load('/home/rzhu/Desktop/projects/kinase_analysis/data/abl/clustering/dfg_dihed_centroids.npy', allow_pickle=True).item()
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


# def assign_dfg_dihed(dbdihed, spatial_assignment, centroids=None, noise_cutoff=1):
#     """
#     Assign a SINGLE conformation to a one of the seven dihedral clusters based on the distance to the cluster's centroid.
#     Cluster indices:   -1 : noise,
#                         --- DFG-in ---
#                         0 : BLAminus,
#                         1 : BLAplus,
#                         2 : ABAminus,
#                         3 : BLBminus,
#                         4 : BLBplus,
#                         5 : BLBtrans
#                         --- DFG-inter ---
#                         6 : BABtrans
#                         --- DFG-out ---
#                         7 : BBAminus}
#     """

#     if centroids is None:
#         centroids = np.load('/home/rzhu/Desktop/projects/kinase_analysis/data/abl/cluster_centers/dfg_dihed_centroids.npy', allow_pickle=True).item()
#     centroids = centroids[['dfg-in', 'dfg-inter', 'dfg-out'][spatial_assignment]]
#     distance = np.array([dihedral_distance(dbdihed, centroid) for centroid in centroids])

#     if np.min(distance) > noise_cutoff:
#         return -1
#     elif spatial_assignment == 0:  # DFG-in group indexing from 0
#         return np.argmin(distance)
#     elif spatial_assignment == 1:  # DFG-inter group indexing from 6
#         return np.argmin(distance) + 6
#     elif spatial_assignment == 2:  # DFG-out group indexing from 7
#         return np.argmin(distance) + 7
    

def dfg_featuriser(traj, protein, save_to_disk=None, 
                   confidence_threshold=0.01, dihed_centroids=None, dihed_cutoff=1, 
                   sptial_name='dist_group', dihed_name='dihed_group') -> list[np.ndarray]:
    '''
    Compute the dfg dihedral group assignment to Abl structures based on Dunbrack distances and dihedrals.
    It can be a trajectory or a group of conformations. 
    '''
    
    dists = dbdist_featuriser(traj, protein)
    diheds = dbdihed_featuriser(traj, protein)[:, [0,1,2,3,4,5,8]]

    assert dists.shape[0] == diheds.shape[0], 'Feature vectors should have the same length'
    assert dists.shape[1] == 2, 'Incorrect number of distance features, should be 2'
    assert diheds.shape[1] == 7, 'Incorrect number of dihedral features, should be 7'

    spatial_assignments = assign_dfg_spatial_hdbscan(dists, protein, confidence_threshold)
    dihed_assignments = np.array(list(map(lambda ds: assign_dfg_dihed(ds[0], ds[1], dihed_centroids, dihed_cutoff), zip(diheds, spatial_assignments))))
    if save_to_disk is not None: 
        dir = save_to_disk.parent
        stem = save_to_disk.stem.split('_')[0]
        np.save(dir/f'{stem}_{sptial_name}.npy', spatial_assignments)
        np.save(dir/f'{stem}_{dihed_name}.npy', dihed_assignments)

    return spatial_assignments, dihed_assignments


# def dfg_featuriser(dists, diheds, dist_centroids=None, dihed_centroids=None, dihed_cutoff=1, save_to_disk=None) -> list[np.ndarray]:
#     '''
#     Compute the dfg dihedral group assignment to Abl structures based on Dunbrack distances and dihedrals.
#     It can be a trajectory or a group of conformations. 
#     '''
    
#     assert dists.shape[0] == diheds.shape[0], 'Feature vectors should have the same length'
#     assert dists.shape[1] == 2, 'Incorrect number of distance features, should be 2'
#     assert diheds.shape[1] == 7, 'Incorrect number of dihedral features, should be 7'

#     spatial_assignments = assign_dfg_spatial(dists, dist_centroids)
#     dihed_assignments = np.array(list(map(lambda ds: assign_dfg_dihed(ds[0], ds[1], dihed_centroids, dihed_cutoff), zip(diheds, spatial_assignments))))
#     if save_to_disk is not None: 
#         np.save(save_to_disk[0], spatial_assignments)
#         np.save(save_to_disk[1], dihed_assignments)

#     return spatial_assignments, dihed_assignments


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