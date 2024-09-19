import csv
import subprocess
import os
from datetime import datetime
from time import sleep
# Define the input and output file paths
input_file = 'database.txt'
output_file = 'fed_results/512x5_0.25var_fwd_xtb_bs6_T5_beta_weighted.csv'

import os

# Set the LD_PRELOAD environment variable
os.environ['LD_PRELOAD'] = '/usr/lib/x86_64-linux-gnu/libgomp.so.1'

# Function to run the command and capture the output
def run_command(smiles, local_model, output_dir, load_from_most_recent=False):
    command = [
        'python', 'train.py', '--t_scale', '0.05', '--T', '1', '--epochs', '15000',
        '--batch_size', '6', '--energy', 'xtb', '--local_model', local_model,
        '--output_dir',  output_dir, #'--langevin',
       '--learned_variance','--log_var_range', '0.5',# '--learn_pb', '--pb_scale_range', '0.5',
        '--patience', '25000', '--model', 'mlp', #,
       # '--conditional_flow_model',#'--ld_step', '0.01','--ld_schedule',
        '--smiles', smiles, '--temperature', '300', '--zero_init', '--clipping',
        '--pis_architectures', '--mode_fwd', 'tb-avg',#'--mode_bwd', 'tb-avg', '--both_ways',#'--max_iter_ls', '100', '--burn_in', '50',
        '--lr_policy', '1e-3', '--lr_back', '1e-3', '--lr_flow', '1e-3', 
        # '--exploratory', '--exploration_wd', '--exploration_factor', '2.',# '--local_search',
        # '--buffer_size', '600000', '--prioritized', 'rank', '--rank_weight', '0.05',
        # '--target_acceptance_rate', '0.574', '--beta', '5',
        '--hidden_dim', '512', '--joint_layers', '5', '--s_emb_dim', '512',
        '--t_emb_dim', '512', '--harmonics_dim', '512'#, '--plot',
    ]
    # if load_from_most_recent:
    #     command.append('--load_from_most_recent')
    if 'solvation' not in local_model:
        command.append('--torchani-model')
        command.append('weights/torchani-vacuum.pt')
    if 'solvation' in local_model:
        command.append('--solvate')
        command.append('--torchani-model')
        command.append('weights/torchani-solvent.pt')
    print(command)
    subprocess.run(command)

# Function to read the output file and extract the required value
def read_output_file(smiles, local_model, output_dir):
    keyword = ''
    smiles = smiles.replace('/', '_')
    if 'vacuum' in local_model:
        keyword = 'vacuum'
    elif 'solvation' in local_model:
        keyword = 'solvation' 
    output_file = f'temp/{output_dir}/{smiles}_{keyword}.txt'
    with open(output_file, 'r') as f:
        lines = f.readlines()
        logZ, logZlb = None, None
        for line in lines:
            if line.startswith('log_Z:'):
                logZ = line.split(':')[1].strip()
            elif line.startswith('log_Z_lb:'):
                logZlb = line.split(':')[1].strip()
            elif line.startswith('log_Z_lb_std:'):
                logZlb_std = line.split(':')[1].strip()
            elif line.startswith('log_Z_std:'):
                logZ_std = line.split(':')[1].strip()
            elif line.startswith('log_Z_learned:'):
                logZ_learned = line.split(':')[1].strip()
            elif line.startswith('log_Z_learned_std:'):
                logZ_learned_std = line.split(':')[1].strip()
    return logZ, logZlb, logZ_std, logZlb_std, logZ_learned, logZ_learned_std

# Check if the output file already exists and read existing results
existing_results = {}
if os.path.exists(output_file):
    with open(output_file, 'r') as outfile:
        reader = csv.reader(outfile)
        next(reader)  # Skip header
        for row in reader:
            smiles = row[0]
            existing_results[smiles] = row

# Read the input file and process each line
with open(input_file, 'r') as infile, open(output_file, 'a', newline='') as outfile:
    reader = csv.reader(infile, delimiter=';')
    writer = csv.writer(outfile)
    
    # create unique ID for output directory
    output_dir = str(datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    # Write the header if the file is new
    if not existing_results:
        writer.writerow(['SMILES', 'experimental_val', 'fed_Z_learned', 'fed_Z', 'fed_Z_lb', 'timestamp'])

    for row in reader:
        if row[0].startswith('#'):
            continue  # Skip header or comment lines

        smiles = row[1].strip()
        experimental_val = row[3]
        experimental_uncertainty = row[4]

        # Skip SMILES that have already been processed
        if smiles in existing_results:
            continue
        
        # species = ['Br', 'P', 'I'] 
        # if any(s in species for s in smiles.strip()):
        #     print('Skipping', smiles)
        #     continue

        local_model_vacuum = 'weights/egnn_vacuum_small_with_hs_final'
        local_model_solvation = 'weights/egnn_solvation_small_with_hs_final'

        run_command(smiles, local_model_vacuum, output_dir)
        print('Vacuum done')
        run_command(smiles, local_model_solvation, output_dir, load_from_most_recent=True)
        print('Solvation done')

        # Read the output files
        logZ_vacuum, logZlb_vacuum, logZ_std_vacuum, logZlb_std_vacuum, logZ_learned_vacuum, logZ_learned_std_vacuum = read_output_file(smiles, local_model_vacuum, output_dir)
        logZ_solvation, logZlb_solvation, logZ_std_solvation, logZlb_std_solvation, logZ_learned_solvation, logZ_learned_std_solvation = read_output_file(smiles, local_model_solvation, output_dir)

        try:
            float(logZ_vacuum)
        except ValueError:
            print('Error in parsing logZs, can\'t convert to float')
            continue
        # Calculate fed_Z and fed_Z_lb
        fed_Z = float(logZ_solvation) - float(logZ_vacuum)
        fed_Z_lb = float(logZlb_solvation) - float(logZlb_vacuum) 
        fed_Z_learned = float(logZ_learned_solvation) - float(logZ_learned_vacuum)

        # Calculate fed uncertainty
        fed_uncertainty = (float(logZ_std_vacuum)**2 + float(logZ_std_solvation)**2)**0.5
        fed_uncertainty_lb = (float(logZlb_std_vacuum)**2 + float(logZlb_std_solvation)**2)**0.5
        fed_uncertainty_learned = (float(logZ_learned_std_vacuum)**2 + float(logZ_learned_std_solvation)**2)**0.5

        # Round values to the third significant digit
        fed_Z = f"{fed_Z:.3g} ± {fed_uncertainty:.3g}"
        fed_Z_lb = f"{fed_Z_lb:.3g} ± {fed_uncertainty_lb:.3g}"
        fed_Z_learned = f"{fed_Z_learned:.3g} ± {fed_uncertainty_learned:.3g}"

        # Get the current timestamp
        timestamp = datetime.now().strftime('%d-%m-%Y %H-%M')

        # Write the results to the CSV file
        writer.writerow([smiles, experimental_val+' ± '+experimental_uncertainty, fed_Z_learned, fed_Z, fed_Z_lb, timestamp])
        outfile.flush()

print("Processing complete. Results saved to", output_file)
