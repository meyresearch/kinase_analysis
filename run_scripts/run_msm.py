from pathlib import Path
import pandas as pd
import argparse
from MSMEstimation import MSMEstimation
from TrajData import TrajData


def parse_dataset(traj_set_str):
    """
    Parse a single set of (key, rtraj_dir, ftraj_dir, dt) from a string.
    Input format: key, rtraj_dir,ftraj_dir,dt (comma-separated)
    """
    try:
        key, rtraj_dir, ftraj_dir, dt = traj_set_str.split(",")
        return str(key.strip()), Path(rtraj_dir.strip()), Path(ftraj_dir.strip()), float(dt.strip())
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Each trajectory set must be formatted as 'rtraj_dir,ftraj_dir,dt'"
        )


def main():
    parser = argparse.ArgumentParser(description="Run MSM studies with configurable paths.")
    parser.add_argument(
        "--protein", 
        type=str, 
        required=True, 
        help="Name of the protein."
    )
    parser.add_argument(
        "--hps_csv", 
        type=Path, 
        required=True, 
        help="Path to the hyperparameter CSV file."
    )
    parser.add_argument(
        "--hp_idx", 
        nargs="+", 
        type=int, 
        required=False,
        default=None, 
        help="List of hyperparameter indices to run."
    )
    parser.add_argument(
        "--wk_dir", 
        type=Path, 
        required=True, 
        help="Path to the working directory for output."
    )
    parser.add_argument(
        "--datasets", 
        nargs="+", 
        type=parse_dataset, 
        required=True, 
        help=(
            "Sets of trajectory data as (key,rtraj_dir,ftraj_dir,dt). "
            "Provide multiple sets separated by space, e.g., "
            "'key_0, dir_to_raw_trajs_0, dir_to_feature_trajs_0, 0.05' 'key_1, dir_to_raw_trajs_1, dir_to_feature_trajs_1, 1'"
        )
    )
    args = parser.parse_args()

    # if hp_idx is None, run all hyperparameters in the hps table
    hps_df = pd.read_csv(args.hps_csv)
    if args.hp_idx is None:
        hp_indices = hps_df.hp_id.to_list()
    else:
        hp_indices = args.hp_idx

    # Create TrajData object and load datasets
    traj_data = TrajData(protein=args.protein)
    for key, rtraj_dir, ftraj_dir, dt in args.traj_sets:
        traj_data.add_dataset(
            key=key,
            rtraj_dir=rtraj_dir, 
            ftraj_dir=ftraj_dir, 
            dt=dt
        )

    # Create MSMEstimation object and run studies
    msm_est = MSMEstimation(
        hps_table=hps_df,
        traj_data=traj_data,
        wk_dir=args.wk_dir
    )

    msm_est.run_studies(hp_indices=hp_indices)


if __name__ == '__main__':
    main()