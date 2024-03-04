from pathlib import Path
from natsort import natsorted

raw_traj_dir = Path('/arc/EGFR/')
raw_traj_files = natsorted([traj for traj in raw_traj_dir.glob('run*-clone?.h5')])

traj_dir = Path('/arc/egfr_processed')
traj_files = natsorted([traj for traj in traj_dir.rglob('run*-clone?.h5')])

ftraj_dir = Path('data_egfr/ftrajs')
ftraj_dist_files = natsorted([ftraj for ftraj in ftraj_dir.rglob('run*-clone?_dbdist.npy')])
ftraj_dihed_files = natsorted([ftraj for ftraj in ftraj_dir.rglob('run*-clone?_dbdihed.npy')])
ftraj_aloop_files = natsorted([ftraj for ftraj in ftraj_dir.rglob('run*-clone?_aloop.npy')])
ftraj_ploop_files = natsorted([ftraj for ftraj in ftraj_dir.rglob('run*-clone?_ploop.npy')])
ftraj_achelix_files = natsorted([ftraj for ftraj in ftraj_dir.rglob('run*-clone?_achelix.npy')])
ftraj_rspine_files = natsorted([ftraj for ftraj in ftraj_dir.rglob('run*-clone?_rspine.npy')])

assert len(traj_files) == len(ftraj_dist_files) == len(ftraj_dihed_files)== len(ftraj_aloop_files) \
    == len(ftraj_ploop_files) == len(ftraj_achelix_files) == len(ftraj_rspine_files), \
         'The number of trajectories and their features do not match'