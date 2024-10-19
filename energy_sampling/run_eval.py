import subprocess
import os
import argparse

# Create argument parser
parser = argparse.ArgumentParser(description='Run eval.py for multiple SMILES')
parser.add_argument('--directory', type=str, required=True, help='Path to the directory containing SMILES subdirectories')

# Parse command line arguments
args = parser.parse_args()
directory = args.directory

# List of SMILES strings (subdirectory names)
smiles_list = [name for name in os.listdir(directory) if os.path.isdir(os.path.join(directory, name))]

# Fixed parameters
common_args = [
    "--eval", "4096",
    "--epochs", "10000",
    "--batch_size", "6",
    "--energy", "xtb",
    "--patience", "25000",
    "--model", "mlp",
    "--temperature", "300",
    "--zero_init",
    "--clipping",
    "--pis_architectures",
    "--mode_fwd", "tb-avg",
    "--lr_policy", "1e-4",
    "--lr_back", "1e-4",
    "--lr_flow", "1e-3",
    "--hidden_dim", "512",
    "--joint_layers", "5",
    "--s_emb_dim", "512",
    "--t_emb_dim", "512",
    "--harmonics_dim", "512",
    "--t_scale", "0.01",
    '--learned_variance','--log_var_range', '0.01',
    "--T", "1",
    "--local_model", "weights/egnn_solvation_small_with_hs_final",
]


for smiles in smiles_list:
    args_list = [
        "python", "eval.py",
        "--smiles", smiles,
    ] + common_args

    print("Running eval.py with SMILES:", smiles)
    subprocess.run(args_list)
