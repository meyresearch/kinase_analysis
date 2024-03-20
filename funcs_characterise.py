### Functions to calculate properties of conformations or clusters of conformations.

from typing import *
import numpy as np
import MDAnalysis as mda
import mdtraj as md
from pathlib import Path
import os


def cal_within_state_rmsd(sample, atom_selection='mass>1.1 and backbone') -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the residue-wise RMSD wihtin a state
    """
    
    no_res = len([_ for _ in sample.top.residues])
    no_frame = len(sample)

    rmsd_mean = []
    rmsd_std = []

    for residue in range(no_res):
        xyz = sample.atom_slice(sample.top.select(f"resid {residue} and {atom_selection}")).xyz
        rmsd_matrix = np.zeros((no_frame, no_frame))

        for i in range(no_frame):
            for j in range(no_frame):
                if j > i: rmsd_matrix[i, j] = np.sqrt(np.mean((xyz[i,:,:] - xyz[j,:,:])**2))
                else: rmsd_matrix[i, j] = rmsd_matrix[j, i]

        ltri = np.tril(rmsd_matrix, -1)
        nonzero_ltri = ltri[np.nonzero(ltri)]
        rmsd_mean.append(np.mean(nonzero_ltri))
        rmsd_std.append(np.std(nonzero_ltri))

    return np.array(rmsd_mean), np.array(rmsd_std)


def cal_between_states_rmsd(sample_a, sample_b, atom_selection='mass>1.1 and backbone') -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the residue-wise RMSD between two states 
    """

    no_res_a = len([_ for _ in sample_a.top.residues])
    no_res_b = len([_ for _ in sample_b.top.residues])
    assert no_res_a == no_res_b, "The number of residues in the two samples are not the same."

    no_frame_a = len(sample_a)
    no_frame_b = len(sample_b)

    rmsd_mean = []
    rmsd_std = []

    for residue in range(no_res_a):
        xyz_a = sample_a.atom_slice(sample_a.top.select(f"resid {residue} and {atom_selection}")).xyz
        xyz_b = sample_a.atom_slice(sample_a.top.select(f"resid {residue} and {atom_selection}")).xyz

        rmsd_matrix = np.zeros((no_frame_a, no_frame_b))

        for i in range(no_frame_a):
            for j in range(no_frame_b):
                rmsd_matrix[i, j] = np.sqrt(np.mean((xyz_a[i,:,:] - xyz_b[j,:,:])**2))

        rmsd_mean.append(np.mean(rmsd_matrix))
        rmsd_std.append(np.std(rmsd_matrix))

    return np.array(rmsd_mean), np.array(rmsd_std)


def cal_between_states_diff(mean_AB, std_A, std_B, normalise=True):
    """
    Scale the residue-wise RMSD between two states by standard deviations
    """

    score = np.array(mean_AB)/(np.array(std_A)*np.array(std_B))
    if normalise: score = (score-np.min(score))/(np.max(score)-np.min(score))
    return score


def save_example_with_property(ref_path, property, save_path):
    """
    Save the reference structure with the a computed property as the B-factors
    """

    ref = mda.Universe(ref_path)
    for residue, p in zip(ref.residues, property):
        residue.atoms.tempfactors = p
    with mda.Writer(save_path, multiframe=False) as W:
        W.write(ref)