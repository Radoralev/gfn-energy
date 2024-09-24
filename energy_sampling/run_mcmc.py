import argparse
import os
import numpy as np
from energies import MoleculeFromSMILES_XTB
from utils import logmeanexp
from cobaya.run import run
from openmm import unit
from tqdm import tqdm
import torch

# Constants
T = 298.15  # Temperature in Kelvin
kB = unit.BOLTZMANN_CONSTANT_kB.value_in_unit(unit.hartree / unit.kelvin)
hartree_to_kcal = 627.509  # Conversion factor from Hartree to kcal/mol
beta = 1 / (kB * T)  # Inverse temperature

def setup_mc(energy, n_dims, min_max=(-np.pi, np.pi), max_samples=1):
    # Generate dynamic angle names based on n_dims
    angle_names = [f"angle_{i}" for i in range(n_dims)]
    
    # Define the likelihood function accepting **kwargs
    def likelihood(**kwargs):
        # Extract current input values
        current_input_values = [kwargs[p] for p in angle_names]
        # Compute the log probability using the energy object
        logp = energy.log_reward(torch.tensor(current_input_values).unsqueeze(0))
        # Optionally compute derived parameters
        # derived = {"sum_angles": sum(current_input_values)}
        # return logp, derived
        return logp.item()
    
    # Cobaya configuration
    config = {
        "likelihood": {
            "my_likelihood": {
                "external": likelihood,
                "input_params": angle_names,
                # "output_params": ["sum_angles"],  # If you have derived parameters
            }
        },
        "params": {
            p: {
                "prior": {"min": min_max[0], "max": min_max[1]},
                "ref": 0,
                "proposal": 0.02
            }
            for p in angle_names
        },
        "sampler": {
            "mcmc": {
                "max_samples": max_samples,
            }
        },
    }
    
    # Run the Cobaya MCMC sampler with the defined config
    updated_info, sampler = run(config)
    print("Updated information: ", updated_info)
    samples = sampler.products()['sample'][angle_names].values
    return samples, sampler, angle_names

def run_mc(sampler, n_samples, angle_names):
    rejected = []
    accepted = 0  # Counter for accepted samples
    
    with tqdm(total=n_samples, desc="Sampling") as pbar:
        for i in range(n_samples):
            trial = sampler.current_point.values.copy()
            sampler.proposer.get_proposal(trial)
            trial_results = sampler.model.logposterior(trial)
            accept = sampler.metropolis_accept(trial_results.logpost, sampler.current_point.logpost)
            
            if accept:
                accepted += 1
            else:
                rejected.append(trial)
    
            sampler.process_accept_or_reject(accept, trial, trial_results)
            
            pbar.set_postfix({
                'Acceptance Rate': f"{accepted / (i + 1):.2%}",
                'Accepted Samples': accepted
            })
            pbar.update(1)
    
    samples = sampler.products()['sample'][angle_names].values
    return rejected, samples

def main():
    parser = argparse.ArgumentParser(description="MCMC sampling script with SMILES input and solvation options.")
    
    parser.add_argument('--smiles', type=str, required=True, help="The SMILES string of the molecule.")
    parser.add_argument('--solvate', action='store_true', help="Specify if solvation is to be considered.")
    parser.add_argument('--samples', type=int, default=1000, help="Number of accepted samples to collect.")
    parser.add_argument('--output_dir', type=str, required=True, help="Output directory to save rejected and accepted samples as numpy arrays.")
    
    args = parser.parse_args()

    # Generate energies based on the SMILES and solvation state
    if args.solvate:
        energy = MoleculeFromSMILES_XTB(args.smiles, temp=T, solvate=True)
    else:
        energy = MoleculeFromSMILES_XTB(args.smiles, temp=T, solvate=False)
    
    # Get the number of dimensions (angles)
    n_dims = energy.data_ndim
    
    # Setup and run MCMC
    samples, sampler, angle_names = setup_mc(energy, n_dims, max_samples=1)
    rejected, samples = run_mc(sampler, args.samples, angle_names)

    # Ensure the output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Save rejected and accepted samples as numpy arrays
    np.save(os.path.join(args.output_dir, f'rejected_samples_{args.solvate}_mcmc.npy'), np.array(rejected))
    np.save(os.path.join(args.output_dir, f'accepted_samples_{args.solvate}_mcmc.npy'), np.array(samples))
    
    print('Rejected samples saved:', len(rejected))
    print('Accepted samples saved:', len(samples))

if __name__ == "__main__":
    main()