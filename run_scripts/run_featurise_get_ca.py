import sys
sys.path.insert(0, '/home/rzhu/Desktop/projects/kinase_analysis/src/')
from funcs_featurise import *
from TrajData import TrajData

traj_data = TrajData(protein='abl')
traj_data.add_dataset(                                                                                                                                                                              
    key='abl-pdb-50ps',
    rtraj_dir='/arc/abl_processed/',
    rtraj_glob='*.h5', 
    ftraj_dir='/home/rzhu/Desktop/projects/kinase_analysis/data/abl/abl-pdb-50ps/ftrajs', 
    dt=0.05
)
traj_data.featurize(
    key='abl-pdb-50ps', 
    feature_names=['ca-1ns-all'],
    featurisers=[ca_coords_featuriser],
    delta_ca=1, 
    delta_t=20
    )
