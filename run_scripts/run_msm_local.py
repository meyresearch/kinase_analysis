from pathlib import Path
import pandas as pd
from MSMEstimation import MSMEstimation
from TrajData import TrajData


hps_df = pd.read_csv('../data/abl/msm/markov_priors/hps.csv')

# Create TrajData object and load datasets
traj_data = TrajData(protein='abl')
traj_data.add_dataset(                                                                                                                                                                              
    key='abl-pdb-1ns',
    rtraj_dir='.', 
    ftraj_dir='/home/rzhu/Desktop/projects/kinase_analysis/data/abl/ftrajs', 
    dt=1
)

# Create MSMEstimation object and run studies
msm_est = MSMEstimation(
    hps_table=hps_df,
    traj_data=traj_data,
    wk_dir='/home/rzhu/Desktop/projects/kinase_analysis/data/abl/msm/markov_priors'
)

msm_est.run_studies(hp_indices=[1,2,3,4,5,6])