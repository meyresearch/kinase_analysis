import sys
sys.path.insert(0, '/home/rzhu/Desktop/projects/kinase_analysis/src/')
from funcs_featurise import *
from funcs_db_assign import dfg_featuriser
from TrajData import TrajData

# traj_data = TrajData(protein='met-af2')
# traj_data.add_dataset(
#     key='met-af2-50ps',
#     rtraj_dir='/arc/met-af2-50ps_processed/',
#     rtraj_glob='*.h5', 
#     ftraj_dir='/home/rzhu/Desktop/projects/kinase_analysis/data/met/met-af2-50ps/ftrajs', 
#     dt=0.05
# )
# traj_data.add_dataset(
#     key='met-af2-1ns',
#     rtraj_dir='/arc/met-af2-1ns_processed/',
#     rtraj_glob='*.h5', 
#     ftraj_dir='/home/rzhu/Desktop/projects/kinase_analysis/data/met/met-af2-1ns/ftrajs', 
#     dt=1
# )

# traj_data.featurize(
#     key='met-af2-50ps',
#     featurisers=[dfg_featuriser],
#     top=None,
#     confidence_threshold=0.01, dihed_cutoff=1,
#     hdbscan_model = '/home/rzhu/Desktop/projects/kinase_analysis/data/abl/clustering/hdbscan.pkl',
#     dihed_centroids = '/home/rzhu/Desktop/projects/kinase_analysis/data/abl/clustering/dfg_dihed_centroids.npy',
#     spatial_name='hdbscan_dist_group', dihed_name='hdbscan_dihed_group',
# )
# traj_data.featurize(
#     key='met-af2-1ns',
#     featurisers=[dfg_featuriser],
#     top=None,
#     confidence_threshold=0.01, dihed_cutoff=1,
#     hdbscan_model = '/home/rzhu/Desktop/projects/kinase_analysis/data/abl/clustering/hdbscan.pkl',
#     dihed_centroids = '/home/rzhu/Desktop/projects/kinase_analysis/data/abl/clustering/dfg_dihed_centroids.npy',
#     spatial_name='hdbscan_dist_group', dihed_name='hdbscan_dihed_group',
# )

traj_data = TrajData(protein='met-pdb')
traj_data.add_dataset(
    key='met-pdb-50ps',
    rtraj_dir='/arc/met-pdb-50ps_processed/',
    rtraj_glob='*.h5', 
    ftraj_dir='/home/rzhu/Desktop/projects/kinase_analysis/data/met/met-pdb-50ps/ftrajs', 
    dt=0.05
)
traj_data.add_dataset(
    key='met-pdb-1ns',
    rtraj_dir='/arc/met-pdb-1ns_processed/',
    rtraj_glob='*.h5', 
    ftraj_dir='/home/rzhu/Desktop/projects/kinase_analysis/data/met/met-pdb-1ns/ftrajs', 
    dt=1
)

traj_data.featurize(
    key='met-pdb-50ps',
    featurisers=[dfg_featuriser],
    top=None,
    confidence_threshold=0.01, dihed_cutoff=1, mapping={-1: -1, 0: 2, 1: 0, 2: 1}, 
    hdbscan_model = '/home/rzhu/Desktop/projects/kinase_analysis/data/abl/clustering/hdbscan.pkl',
    dihed_centroids = '/home/rzhu/Desktop/projects/kinase_analysis/data/abl/clustering/dfg_dihed_centroids.npy',
    spatial_name='hdbscan_dist_group', dihed_name='hdbscan_dihed_group',
)
traj_data.featurize(
    key='met-pdb-1ns',
    featurisers=[dfg_featuriser],
    top=None,
    confidence_threshold=0.01, dihed_cutoff=1, mapping={-1: -1, 0: 2, 1: 0, 2: 1}, 
    hdbscan_model = '/home/rzhu/Desktop/projects/kinase_analysis/data/abl/clustering/hdbscan.pkl',
    dihed_centroids = '/home/rzhu/Desktop/projects/kinase_analysis/data/abl/clustering/dfg_dihed_centroids.npy',
    spatial_name='hdbscan_dist_group', dihed_name='hdbscan_dihed_group',
)
