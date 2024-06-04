from mdtraj.formats.hdf5 import HDF5TrajectoryFile
import mdtraj as md

import numpy as np
from tqdm import tqdm
from pathlib import Path
import os
from natsort import natsorted
import re
import builtins

original_print = builtins.print
def print(*args, **kwargs):
    kwargs.setdefault('flush', True)  # Set flush to True unless it's already specified
    original_print(*args, **kwargs)
builtins.print = print


# Directory containing the raw trajectories to be processed
raw_traj_dir = Path('/arc/ABL1/')
raw_traj_files = natsorted([traj for traj in raw_traj_dir.glob('run*-clone?.h5')])
max_run_no = max([int(re.search(r'run([0-9]+)-clone[0-9]+\.h5', f.name).group(1)) for f in raw_traj_files])

# Directory to save the processed trajectories
processed_traj_directory = Path('/arc/abl_processed')
processed_traj_directory.mkdir(exist_ok=True)

# Directory containing the correct topology for each run
topology_paths = natsorted([pdb for pdb in Path('/arc/abl_equilibrated_strucs').glob('RUN*_solute_equilibrated.pdb')])
topology_selection = 'chainid 0 and not name CL and not name NA'


if __name__ == "__main__":
    for i in tqdm(range(max_run_no+1), total=max_run_no+1):
        trajs = natsorted([p for p in raw_traj_dir.rglob(f'run{i}-clone?.h5')])
        print(f"Number of clones to process for run {i}: {len(trajs)}")

        # Subset the correct topology for a specific run 
        assert str(topology_paths[i].stem).split('_')[0] == str(trajs[0].stem).split('-')[0].upper()
        top = md.load(topology_paths[i]).topology
        atom_indices = top.select(topology_selection)
        trajectory_top = top.subset(atom_indices)
        print("Number of selected atoms: ", len(atom_indices))
        
        for traj in trajs:
            if processed_traj_directory.joinpath(f'{traj.stem}.h5').exists():
                print(f"Trajectory {traj.stem} already processed. Skipping...")
                continue
            
            print("Processing trajectory: ", traj.stem)
            processed_traj_file = processed_traj_directory.joinpath(f'{traj.stem}.h5')
            try:
                no_trajectory_coordinates = md.load_frame(traj, 0).xyz.shape[1]
            except Exception as err:
                print(f"Error processing trajectory {traj.stem}. Skipping...")
                print("\nError message: \n", err)
                continue

            # If more atoms in the topology than in the trajectory
            # Remove the last few atoms&bonds (H of the NME) as they won't affect the later analysis
            if sum(1 for _ in trajectory_top.atoms) > no_trajectory_coordinates:
                print(f"Number of trajectory atoms ({no_trajectory_coordinates}) less than topology atoms. Trimming topology atoms.")
                for i in range(sum(1 for _ in trajectory_top.atoms) - no_trajectory_coordinates):
                    atom_indices = np.delete(atom_indices, -1)
                trajectory_top = top.subset(atom_indices)

            # Write file
            trj_file = HDF5TrajectoryFile(processed_traj_file, mode='a')
            trj_file.topology = trajectory_top
            try:
                for chunk in md.iterload(traj, top=trajectory_top, atom_indices=atom_indices, chunk=10000):
                    trj_file.write(coordinates=chunk.xyz, cell_lengths=chunk.unitcell_lengths, cell_angles=chunk.unitcell_angles, time=chunk.time)
                trj_file.close()
            except Exception as err:
                print(f"Error processing trajectory {traj.stem}. Skipping...")
                print("\nError message: \n", err)
                trj_file.close()
                if os.path.exists(processed_traj_file): os.remove(processed_traj_file)
                continue