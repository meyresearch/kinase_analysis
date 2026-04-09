import sys
sys.path.insert(0, '/home/rzhu/Desktop/projects/kinase_analysis/src/')
from funcs_featurise import *
from TrajData import TrajData

traj_data = TrajData(protein='abl')
dataset_key = 'abl-pdb-50ps'
traj_data.add_dataset(                                                                                                                                                                              
    key=dataset_key,
    rtraj_dir='/arc/abl_processed/',
    rtraj_glob='*.h5', 
    ftraj_dir=f'/home/rzhu/Desktop/projects/kinase_analysis/data/abl/{dataset_key}/ftrajs', 
    dt=0.05
)
traj_data.featurize(
    key=dataset_key, 
    featurisers=[dbdist_featuriser, dbdihed_featuriser, achelix_featuriser, aloop_featuriser]
)