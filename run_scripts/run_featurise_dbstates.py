import sys
sys.path.insert(0, '/home/rzhu/Desktop/projects/kinase_analysis/src/')
from funcs_featurise import *
from funcs_db_assign import dfg_featuriser
from TrajData import TrajData

traj_data = TrajData(protein='abl')
traj_data.add_dataset(                                                                                                                                                                              
    key='abl-pdb-50ps',
    rtraj_dir='/arc/abl_processed/',
    rtraj_glob='*.h5', 
    ftraj_dir='/home/rzhu/Desktop/projects/kinase_analysis/data/abl/ftrajs', 
    dt=0.05
)
traj_data.featurize(
    key='abl-pdb-50ps', 
    featurisers=[dfg_featuriser],
    top=None,
    confidence_threshold=0.01, dihed_centroids=None, dihed_cutoff=1,
    sptial_name='hdbscan_dist_group', dihed_name='hdbscan_dihed_group'
)
