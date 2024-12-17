import sys
sys.path.insert(0, '/home/rzhu/Desktop/projects/kinase_analysis/src/')
from funcs_featurise import *
from TrajData import TrajData

traj_data = TrajData(protein='abl')
traj_data.add_dataset(                                                                                                                                                                              
    key='abl-pdb-1ns',
    rtraj_dir='/arc/abl_processed/',
    rtraj_glob='*.h5', 
    ftraj_dir='/home/rzhu/Desktop/projects/kinase_analysis/data/abl/abl-pdb-1ns/ftrajs', 
    dt=1
)
traj_data.featurize(
    key='abl-pdb-1ns', 
    featurisers=[dbdist_featuriser, dbdihed_featuriser, achelix_featuriser, aloop_featuriser]
)
