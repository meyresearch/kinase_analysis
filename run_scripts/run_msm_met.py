import pandas as pd
import sys
sys.path.insert(0, '/home/rzhu/Desktop/projects/kinase_analysis/src/')
from MSMEstimation import MSMEstimation
from TrajData import TrajData

hps_df = pd.read_csv('hps_example.csv')

# Create TrajData object and load datasets
traj_data = TrajData(protein='met')
traj_data.add_dataset(                                                                                                                                                                              
    key='met-pdb-50ps',
    rtraj_dir='/arc/met-pdb-50ps_processed', 
    ftraj_dir='/home/rzhu/Desktop/projects/kinase_analysis/data/met/met-pdb-50ps/ftrajs', 
    dt=0.05
)
traj_data.add_dataset(                                                                                                                                                                              
    key='met-af2-50ps',
    rtraj_dir='/arc/met-af2-50ps_processed', 
    ftraj_dir='/home/rzhu/Desktop/projects/kinase_analysis/data/met/met-af2-50ps/ftrajs', 
    dt=0.05
)

# Create MSMEstimation object and run studies
msm_est = MSMEstimation(
    hps_table=hps_df,
    traj_data=traj_data,
    wk_dir='/home/rzhu/Desktop/projects/kinase_analysis/data/met/msm/with_prior'
)

msm_est.run_studies(hp_indices=[1,2,3,4])