import argparse
import os
import csv
import numpy as np
from energies import MoleculeFromSMILES_XTB
from utils import logmeanexp
from cobaya.run import run
from cobaya.log import LoggedError
from openmm import unit
from tqdm import tqdm
import torch
from pymbar import other_estimators
from scipy.special import logsumexp
from mcmc_eval_utils import weighted_EXP, compute_weights, fed_estimate_Z, calc_ESS


# Constants
room_temp = 298.15  # Temperature in Kelvin
high_temp = 1000  # High temperature in Kelvin
kB = unit.BOLTZMANN_CONSTANT_kB.value_in_unit(unit.hartree / unit.kelvin)
hartree_to_kcal = 627.509  # Conversion factor from Hartree to kcal/mol
beta = 1 #/ (kB * T)  # Inverse temperature

def main():
    parser = argparse.ArgumentParser(description="MCMC sampling script with SMILES input and solvation options.")

    parser.add_argument('--input_csv', type=str, required=True, help="Input CSV file containing SMILES and experimental data.")
    parser.add_argument('--output_csv', type=str, required=True, help="Output CSV file to save results.")
    parser.add_argument('--samples', type=int, default=1000, help="Number of accepted samples to collect per state.")
    parser.add_argument('--max_samples', type=int, default=100000, help="Maximum number of MCMC iterations per state.")
    parser.add_argument('--output_dir', type=str, required=True, help="Output directory to save samples as numpy arrays.")

    args = parser.parse_args()

    # Read processed SMILES from output CSV file (if it exists)
    processed_smiles = set()
    if os.path.exists(args.output_csv):
        with open(args.output_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                processed_smiles.add(row['SMILES'])

    # Open output CSV file for appending
    output_file_exists = os.path.exists(args.output_csv)
    output_file = open(args.output_csv, 'a', newline='')
    writer = None

    # Read input CSV file and process SMILES one by one
    with open(args.input_csv, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            fields = line.split(';')
            if len(fields) < 10:
                continue  # Skip lines that don't have enough fields
            compound_id = fields[0].strip()
            smiles = fields[1].strip()
            iupac_name = fields[2].strip()
            experimental_value = fields[3].strip()
            experimental_uncertainty = fields[4].strip()
            # Skip if SMILES is already processed
            if smiles in processed_smiles:
                print(f"Skipping SMILES {smiles} as it's already processed.")
                continue
            # Process the SMILES
            print(f"Processing SMILES: {smiles}")

            # Generate energies for solvate and vacuum states
            energy_solvate = MoleculeFromSMILES_XTB(smiles, temp=high_temp, solvate=True, n_jobs=16)

            if energy_solvate.data_ndim == 0 or energy_solvate.data_ndim >= 9:
                continue

            energy_vacuum = MoleculeFromSMILES_XTB(smiles, temp=high_temp, solvate=False, n_jobs=16)

            # Run MCMC sampling for both states
            print("Sampling in vacuum state...")
            samples_vacuum_accepted, samples_vacuum_all = run_mcmc_sampling(energy_vacuum, args.samples, args.max_samples)
            print("Sampling in solvate state...")
            samples_solvate_accepted, samples_solvate_all = run_mcmc_sampling(energy_solvate, args.samples, args.max_samples)

            if samples_vacuum_accepted is None or samples_solvate_accepted is None:
                print(f"Error in MCMC sampling for SMILES {smiles}. Skipping...")
                continue
            # Ensure the output directory exists
            # if not os.path.exists(args.output_dir):
            #     os.makedirs(args.output_dir)

            # Save samples
            # np.save(os.path.join(args.output_dir, f'{compound_id}_vacuum_accepted_samples.npy'), samples_vacuum_accepted)
            # np.save(os.path.join(args.output_dir, f'{compound_id}_solvate_accepted_samples.npy'), samples_solvate_accepted)
            # np.save(os.path.join(args.output_dir, f'{compound_id}_vacuum_all_samples.npy'), samples_vacuum_all)
            # np.save(os.path.join(args.output_dir, f'{compound_id}_solvate_all_samples.npy'), samples_solvate_all)

            # Generate energies for solvate and vacuum states
            energy_solvate = MoleculeFromSMILES_XTB(smiles, temp=room_temp, solvate=True, n_jobs=16)
            energy_vacuum = MoleculeFromSMILES_XTB(smiles, temp=room_temp, solvate=False, n_jobs=16)


            # Compute energies and work values for accepted samples
            energies_vv_acc = compute_energies(energy_vacuum, samples_vacuum_accepted)
            energies_vs_acc = compute_energies(energy_vacuum, samples_solvate_accepted)
            energies_sv_acc = compute_energies(energy_solvate, samples_vacuum_accepted)
            energies_ss_acc = compute_energies(energy_solvate, samples_solvate_accepted)

            w_F_acc = (energies_sv_acc - energies_vv_acc) * beta
            w_R_acc = (energies_vs_acc - energies_ss_acc) * beta

            # # Compute energies and work values for all samples
            # energies_vv_all = compute_energies(energy_vacuum, samples_vacuum_all)
            # energies_vs_all = compute_energies(energy_vacuum, samples_solvate_all)
            # energies_sv_all = compute_energies(energy_solvate, samples_vacuum_all)
            # energies_ss_all = compute_energies(energy_solvate, samples_solvate_all)

            # w_F_all = (energies_sv_all - energies_vv_all) * beta
            # w_R_all = (energies_vs_all - energies_ss_all) * beta

            # Function to compute chunked free energy differences
            def compute_chunked_free_energy_differences(w_F, w_R, energies_vv, energies_ss, sample_type):
                print(w_F.shape)
                chunk_size = len(w_F) // 5  # Split into 5 chunks
                n_chunks = len(w_F) // chunk_size 
                chunked_results = []

                for i in range(n_chunks):
                    chunk_w_F = w_F[i*chunk_size:(i+1)*chunk_size]
                    chunk_w_R = w_R[i*chunk_size:(i+1)*chunk_size]
                    chunk_energies_vv = energies_vv[i*chunk_size:(i+1)*chunk_size]
                    chunk_energies_ss = energies_ss[i*chunk_size:(i+1)*chunk_size]

                    chunk_result = compute_free_energy_differences(chunk_w_F, chunk_w_R, chunk_energies_vv, chunk_energies_ss, sample_type)
                    chunked_results.append(chunk_result)

                # Aggregate results
                aggregated_results = {}
                for key in chunked_results[0].keys():
                    values = [chunk[key] for chunk in chunked_results if isinstance(chunk[key], (int, float)) and not np.isnan(chunk[key])]
                    aggregated_results[key] = np.mean(values)
                    aggregated_results[f'{key}_std'] = np.std(values)

                return aggregated_results

            # Compute free energy differences using accepted samples
            results_accepted = compute_chunked_free_energy_differences(w_F_acc, w_R_acc, energies_vv_acc, energies_ss_acc, sample_type='accepted')

            # Compute free energy differences using all samples
            # results_all = compute_chunked_free_energy_differences(w_F_all, w_R_all, energies_vv_all, energies_ss_all, sample_type='all')
            # Merge the results
            results = {**results_accepted }#**results_all}

            # Add SMILES and experimental values to results
            results['SMILES'] = smiles
            results['experimental_value'] = experimental_value
            results['experimental_uncertainty'] = experimental_uncertainty

            # Round float results to 4 decimal places
            for key, value in results.items():
                if isinstance(value, float):
                    results[key] = round(value, 4)

            # Write results to output CSV
            if writer is None:
                # Initialize CSV writer with fieldnames
                fieldnames = ['SMILES', 'experimental_value', 'experimental_uncertainty',
                              'deltaF_EXP_accepted', 'deltaF_EXP_accepted_std', 'deltaF_BAR_accepted', 'deltaF_BAR_accepted_std',
                              'delta_f_GFN_samples_accepted', 'delta_f_GFN_samples_accepted_std',
                              'deltaF_EXP_all', 'deltaF_EXP_all_std', 'deltaF_BAR_all', 'deltaF_BAR_all_std',
                              'delta_f_GFN_samples_all', 'delta_f_GFN_samples_all_std']
                writer = csv.DictWriter(output_file, fieldnames=fieldnames)
                if not output_file_exists:
                    writer.writeheader()
            writer.writerow(results)
            output_file.flush()
            processed_smiles.add(smiles)

    output_file.close()

def run_mcmc_sampling(energy, n_samples_to_collect, max_samples):
    # Generate dynamic angle names based on n_dims
    n_dims = energy.data_ndim
    angle_names = [f"angle_{i}" for i in range(n_dims)]

    # Define the likelihood function accepting **kwargs
    def likelihood(**kwargs):
        # Extract current input values
        current_input_values = [kwargs[p] for p in angle_names]
        # Compute the log probability using the energy object
        logp = energy.log_reward(torch.tensor(current_input_values).unsqueeze(0))
        # Return log probability
        return logp.item()

    # Cobaya configuration
    config = {
        "likelihood": {
            "my_likelihood": {
                "external": likelihood,
                "input_params": angle_names,
            }
        },
        "params": {
            p: {
                "prior": {"min": -np.pi, "max": np.pi},
                "ref": 0,
                "proposal": 0.01
            }
            for p in angle_names
        },
        "sampler": {
            "mcmc": {
                "max_samples": max_samples,
                'Rminus1_stop': 0.02,
            }
        },
    }

    # Run the Cobaya MCMC sampler with the defined config
    try:
        updated_info, sampler = run(config)
    except LoggedError as e:
        print(f"Error in Cobaya run: {e}")
        return None, None
    # Now, we run the sampler until we collect n_samples_to_collect accepted samples
    all_samples = np.array(sampler.products()['sample'][angle_names].values).tolist()
    accepted = len(sampler.products()['sample'][angle_names].values)  # Counter for accepted samples
    total_samples = len(sampler.products()['sample'][angle_names].values)  # Total number of attempted samples

    # with tqdm(total=n_samples_to_collect, desc="Sampling") as pbar:
    #     while accepted < n_samples_to_collect:
    #         trial = sampler.current_point.values.copy()
    #         sampler.proposer.get_proposal(trial)
    #         trial_results = sampler.model.logposterior(trial)
    #         accept = sampler.metropolis_accept(trial_results.logpost, sampler.current_point.logpost)

    #         if accept:
    #             accepted += 1
    #         # Update the sampler state regardless of acceptance
    #         sampler.process_accept_or_reject(accept, trial, trial_results)

    #         # Record the current state of the chain
    #         all_samples.append(trial)

    #         total_samples += 1

    #         pbar.set_postfix({
    #             'Acceptance Rate': f"{accepted / total_samples:.2%}",
    #             'Accepted Samples': accepted
    #         })
    #         pbar.update(accepted - pbar.n)

    if accepted < n_samples_to_collect:
        print(f"Warning: Only {accepted} samples were accepted out of {n_samples_to_collect} requested.")

    accepted_samples = sampler.products()['sample'][angle_names].values
    all_samples = np.array(all_samples)
    return accepted_samples, all_samples

def compute_energies(energy_model, samples):
    # Compute energies for the given samples using the provided energy model
    energies = []
    batch_size = 100  # Adjust as needed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with tqdm(total=len(samples), desc="Computing energies") as pbar:
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i+batch_size]
            batch_tensor = torch.tensor(batch, dtype=torch.float32).to(device)
            with torch.no_grad():
                energy_values = energy_model.energy(batch_tensor).cpu().numpy()
            energies.extend(energy_values)
            pbar.update(len(batch))
    return np.array(energies)

def compute_free_energy_differences(w_F, w_R, energies_vv, energies_ss, sample_type):
    # Compute free energy differences using BAR, EXP, and fedZ methods
    results = {}
    T = room_temp
    try:
        # Compute EXP estimator
        deltaF_EXP_result = other_estimators.exp(w_F)
        deltaF_EXP = deltaF_EXP_result['Delta_f'] * kB * T * hartree_to_kcal
        deltaF_EXP_std = deltaF_EXP_result['dDelta_f'] * kB * T * hartree_to_kcal
        results[f'deltaF_EXP_{sample_type}'] = deltaF_EXP
        # results[f'deltaF_EXP_std_{sample_type}'] = deltaF_EXP_std
    except Exception as e:
        print(f"Error in EXP estimator ({sample_type} samples): {e}")
        results[f'deltaF_EXP_{sample_type}'] = 'nan'
        # results[f'deltaF_EXP_std_{sample_type}'] = 'nan'

    try:
        # Compute BAR estimator
        deltaF_BAR_result = other_estimators.bar(w_F, w_R)
        deltaF_BAR = deltaF_BAR_result['Delta_f'] * kB * T * hartree_to_kcal
        deltaF_BAR_std = deltaF_BAR_result['dDelta_f'] * kB * T * hartree_to_kcal
        results[f'deltaF_BAR_{sample_type}'] = deltaF_BAR
        # results[f'deltaF_BAR_std_{sample_type}'] = deltaF_BAR_std
    except Exception as e:
        print(f"Error in BAR estimator ({sample_type} samples): {e}")
        results[f'deltaF_BAR_{sample_type}'] = 'nan'
        # results[f'deltaF_BAR_std_{sample_type}'] = 'nan'

    try:
        # Compute fedZ estimator (delta_f_GFN_samples)
        delta_f_GFN_samples = fed_estimate_Z(torch.from_numpy(energies_vv), torch.from_numpy(energies_ss), beta_applied=True).numpy() 
        results[f'delta_f_GFN_samples_{sample_type}'] = delta_f_GFN_samples
    except Exception as e:
        print(f"Error in fedZ estimator ({sample_type} samples): {e}")
        results[f'delta_f_GFN_samples_{sample_type}'] = 'nan'

    return results

if __name__ == "__main__":
    main()
