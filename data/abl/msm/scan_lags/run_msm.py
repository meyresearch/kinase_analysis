import pandas as pd
import sys
sys.path.insert(0, '/home/rzhu/Desktop/projects/kinase_analysis/src/')
from MSMEstimation import MSMEstimation
from TrajData import TrajData


hps_df = pd.read_csv('hps.csv')

# Create TrajData object and load datasets
traj_data = TrajData(protein='abl')
traj_data.add_dataset(                                                                                                                                                                              
    key='abl-pdb-50ps',
    rtraj_dir='/arc/abl_processed', 
    ftraj_dir='/home/rzhu/Desktop/projects/kinase_analysis/data/abl/abl-pdb-50ps/ftrajs', 
    dt=0.05
)

# Create MSMEstimation object and run studies
msm_est = MSMEstimation(
    hps_table=hps_df,
    traj_data=traj_data,
    wk_dir='/home/rzhu/Desktop/projects/kinase_analysis/data/abl/msm/scan_lags'
)

msm_est.run_studies(hp_indices=list(range(1,17)))
