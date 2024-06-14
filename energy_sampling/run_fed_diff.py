import csv
import subprocess
import os
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Define the input and output file paths
input_file = 'database.txt'
output_file = 'results.csv'

# Function to run the command and capture the output
def run_command(smiles, local_model):
    command = [
        'python', 'train.py', '--t_scale', '1.', '--T', '100', '--epochs', '5',
        '--batch_size', '32', '--energy', 'neural', '--local_model', local_model,
        '--smiles', smiles, '--temperature', '300', '--zero_init', '--clipping',
        '--pis_architectures', '--mode_fwd', 'tb-avg', '--mode_bwd', 'tb-avg',
        '--lr_policy', '1e-3', '--lr_back', '1e-3', '--lr_flow', '1e-1',
        '--exploratory', '--exploration_wd', '--exploration_factor', '0.1',
        '--buffer_size', '60000', '--prioritized', 'rank', '--rank_weight', '0.01',
        '--ld_step', '0.1', '--ld_schedule', '--target_acceptance_rate', '0.574',
        '--hidden_dim', '64', '--joint_layers', '2', '--s_emb_dim', '64',
        '--t_emb_dim', '64', '--harmonics_dim', '64'
    ]
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
    return logZ, logZlb

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
        writer.writerow(['SMILES', 'experimental_val', 'experimental_uncertainty', 'fed_Z', 'fed_Z_lb', 'logZ_solvation', 'logZlb_solvation', 'logZ_vacuum', 'logZlb_vacuum', 'timestamp'])

    for row in reader:
        if row[0].startswith('#'):
            continue  # Skip header or comment lines

        smiles = row[1]
        experimental_val = row[3]
        experimental_uncertainty = row[4]

        # Skip SMILES that have already been processed
        if smiles in existing_results:
            continue

        # Run the commands in parallel
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = []
            local_model_vacuum = 'weights/egnn_vacuum_batch_size_32'
            local_model_solvation = 'weights/egnn_solvation_batch_size_32'
            futures.append(executor.submit(run_command, smiles, local_model_vacuum))
            futures.append(executor.submit(run_command, smiles, local_model_solvation))

            # Wait for both commands to complete
            for future in as_completed(futures):
                future.result()

        # Read the output files
        logZ_vacuum, logZlb_vacuum = read_output_file(smiles, local_model_vacuum)
        logZ_solvation, logZlb_solvation = read_output_file(smiles, local_model_solvation)

        # Calculate fed_Z and fed_Z_lb
        fed_Z = float(logZ_vacuum) - float(logZ_solvation)
        fed_Z_lb = float(logZlb_vacuum) - float(logZlb_solvation)

        # Round values to the third significant digit
        logZ_vacuum = f"{float(logZ_vacuum):.3g}"
        logZlb_vacuum = f"{float(logZlb_vacuum):.3g}"
        logZ_solvation = f"{float(logZ_solvation):.3g}"
        logZlb_solvation = f"{float(logZlb_solvation):.3g}"
        fed_Z = f"{fed_Z:.3g}"
        fed_Z_lb = f"{fed_Z_lb:.3g}"

        # Get the current timestamp
        timestamp = datetime.now().strftime('%d-%m-%Y %H-%M')

        # Write the results to the CSV file
        writer.writerow([smiles, experimental_val, experimental_uncertainty, fed_Z, fed_Z_lb, logZ_solvation, logZlb_solvation, logZ_vacuum, logZlb_vacuum, timestamp])
        outfile.flush()

print("Processing complete. Results saved to", output_file)