import csv
import subprocess
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import sleep
# Define the input and output file paths
input_file = 'database.txt'
output_file = 'fed_results/tb_no_var_1k_epochs_small_lr1e5.csv'

# Function to run the command and capture the output
def run_command(smiles, local_model):
    command = [
        'python', 'train.py', '--t_scale', '1.', '--T', '10', '--epochs', '1000',
        '--batch_size', '256', '--energy', 'neural', '--local_model', local_model,
       # '--learned_variance', '--log_var_range', '1', 
        '--patience', '25000',
        '--conditional_flow_model',
        '--smiles', smiles, '--temperature', '300', '--zero_init', '--clipping',
        '--pis_architectures', '--mode_fwd', 'tb',# '--mode_bwd', 'tb',
        '--lr_policy', '1e-5', '--lr_back', '1e-5', '--lr_flow', '1e-2', 
       # '--exploratory', '--exploration_wd', '--exploration_factor', '0.1', '--local_search',
       # '--buffer_size', '60000', '--prioritized', 'rank', '--rank_weight', '0.01',
       # '--ld_step', '0.1', '--ld_schedule', '--target_acceptance_rate', '0.574',
        '--hidden_dim', '64', '--joint_layers', '5', '--s_emb_dim', '64',
        '--t_emb_dim', '64', '--harmonics_dim', '64'
    ]
    print(command)
    subprocess.run(command)

# Function to read the output file and extract the required value
def read_output_file(smiles, local_model):
    keyword = ''
    if 'vacuum' in local_model:
        keyword = 'vacuum'
    elif 'solvation' in local_model:
        keyword = 'solvation' 
    output_file = f'temp/{smiles}_{keyword}.txt'
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

    # Write the header if the file is new
    if not existing_results:
        writer.writerow(['SMILES', 'experimental_val', 'fed_Z', 'fed_Z_lb', 'fed_Z_learned', 'timestamp'])

    # Create a ThreadPoolExecutor with a maximum of 8 workers
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = []
        smiles_list = []

        for row in reader:
            if row[0].startswith('#'):
                continue  # Skip header or comment lines

            smiles = row[1]
            experimental_val = row[3]
            experimental_uncertainty = row[4]

            # Skip SMILES that have already been processed
            if smiles in existing_results:
                continue

            smiles_list.append((smiles, experimental_val, experimental_uncertainty))

        # Function to process a single SMILES
        def process_smiles(smiles, experimental_val, experimental_uncertainty):
            local_model_vacuum = 'weights/egnn_vacuum_small'
            local_model_solvation = 'weights/egnn_solvation_small'
            local_futures = []
            with ThreadPoolExecutor(max_workers=1) as executor2:
                local_futures.append(executor2.submit(run_command, smiles, local_model_vacuum))
                sleep(10)
                local_futures.append(executor2.submit(run_command, smiles, local_model_solvation))

            for future in as_completed(local_futures):
                future.result()
            # Read the output files
            logZ_vacuum, logZlb_vacuum, logZ_std_vacuum, logZlb_std_vacuum, logZ_learned_vacuum, logZ_learned_std_vacuum = read_output_file(smiles, local_model_vacuum)
            logZ_solvation, logZlb_solvation, logZ_std_solvation, logZlb_std_solvation, logZ_learned_solvation, logZ_learned_std_solvation = read_output_file(smiles, local_model_solvation)

            # Calculate fed_Z and fed_Z_lb
            fed_Z = float(logZ_vacuum) - float(logZ_solvation)
            fed_Z_lb = float(logZlb_vacuum) - float(logZlb_solvation)
            fed_Z_learned = float(logZ_learned_vacuum) - float(logZ_learned_solvation)

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
            writer.writerow([smiles, experimental_val+' ± '+experimental_uncertainty, fed_Z, fed_Z_lb, fed_Z_learned, timestamp])
            outfile.flush()

        # Submit tasks for processing SMILES
        for smiles, experimental_val, experimental_uncertainty in smiles_list:
            futures.append(executor.submit(process_smiles, smiles, experimental_val, experimental_uncertainty))

        # Ensure all futures are processed
        for future in as_completed(futures):
            future.result()

print("Processing complete. Results saved to", output_file)
