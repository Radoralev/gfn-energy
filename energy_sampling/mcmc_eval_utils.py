
from energies import MoleculeFromSMILES_XTB
from utils import logmeanexp
from cobaya.run import run
from openmm import unit 
from tqdm import tqdm
import numpy as np
import pymbar
import torch

T = 298.15  # Temperature in Kelvin
kB = unit.BOLTZMANN_CONSTANT_kB.value_in_unit(unit.hartree/unit.kelvin)
hartree_to_kcal = 627.509  # Conversion factor from Hartree to kcal/mol
beta = 1/(kB * T)  # Inverse temperature

v_energy = MoleculeFromSMILES_XTB(' CCCCCC(=O)OC', temp=T, solvate=False)
s_energy = MoleculeFromSMILES_XTB(' CCCCCC(=O)OC', temp=T, solvate=True)

def calc_fed(energies, beta_applied=False):
    en = energies if beta_applied else energies*beta
    factor = hartree_to_kcal * kB * T 
    free_energy = -logmeanexp(-en) * factor
    return free_energy


def brute_force_fed(smiles, T, ground_truth, uncertainty, num_batches=100, batch_size=32):
    v_energy = MoleculeFromSMILES_XTB(smiles, temp=T, solvate=False)
    s_energy = MoleculeFromSMILES_XTB(smiles, temp=T, solvate=True)

    default_ta_vals = v_energy.rd_conf.get_freely_rotatable_tas_values()

    torsions = v_energy.tas
    n = len(torsions)

    distr = torch.distributions.VonMises(torch.tensor([0.0]*n), torch.tensor([4.0]*n))
    # create a tensor to store the energies
    samples = distr.sample((num_batches*batch_size,))
    v_energies = torch.zeros(num_batches*batch_size)
    s_energies = torch.zeros(num_batches*batch_size)
    start = 0
    end = batch_size
    for i in tqdm(range(num_batches)):
        batch = samples[start:end]
        v_energies[start:end] = v_energy.energy(batch) 
        s_energies[start:end] = s_energy.energy(batch)

        if (i+1) % 25 == 0:
            print('Batch {}/{}'.format(i+1, num_batches))
            v_free_energy = calc_fed(v_energies[:end])
            s_free_energy = calc_fed(s_energies[:end])
            fed = s_free_energy - v_free_energy
            print('Error for {}: {}'.format(smiles, fed - ground_truth))

        start = end
        end += batch_size

    v_free_energy = calc_fed(v_energies)
    s_free_energy = calc_fed(s_energies)
    fed = s_free_energy - v_free_energy
    return fed, samples

# https://en.wikipedia.org/wiki/Bennett_acceptance_ratio#The_basic_case
def bar_estimate(v_file, s_file, beta, subsample=0):
    v_states = torch.from_numpy(np.load(v_file))
    s_states = torch.from_numpy(np.load(s_file))

    if subsample > 0:
        indices = np.random.choice(len(v_states), subsample, replace=False)
        v_states = v_states[indices]
        s_states = s_states[indices]


    v_energy = MoleculeFromSMILES_XTB(' CCCCCC(=O)OC', temp=T, solvate=False)
    s_energy = MoleculeFromSMILES_XTB(' CCCCCC(=O)OC', temp=T, solvate=True)
    vv_energies = torch.zeros(len(v_states))
    vs_energies = torch.zeros(len(v_states))
    sv_energies = torch.zeros(len(v_states))
    ss_energies = torch.zeros(len(v_states))

    batch_size = 8
    num_batches = len(v_states) // batch_size

    for i in tqdm(range(num_batches)):
        start = i * batch_size
        end = start + batch_size

        vv_energies[start:end] = v_energy.energy(v_states[start:end])
        vs_energies[start:end] = v_energy.energy(s_states[start:end])
        sv_energies[start:end] = s_energy.energy(v_states[start:end])
        ss_energies[start:end] = s_energy.energy(s_states[start:end])

    # Handle the remaining samples if the total number is not divisible by the batch size
    if len(v_states) % batch_size != 0:
        start = num_batches * batch_size
        end = len(v_states)

        vv_energies[start:end] = v_energy.energy(v_states[start:end])
        vs_energies[start:end] = v_energy.energy(s_states[start:end])
        sv_energies[start:end] = s_energy.energy(v_states[start:end])
        ss_energies[start:end] = s_energy.energy(s_states[start:end]) 
    w_F = ((sv_energies - vv_energies)*beta).cpu().numpy()
    w_R = ((vs_energies - ss_energies)*beta).cpu().numpy()


    exp_avg_f = pymbar.other_estimators.exp(w_F)
    exp_avg_b = pymbar.other_estimators.exp(w_R)

    result = pymbar.other_estimators.bar(w_F, w_R,
                                maximum_iterations=1000000, 
                                iterated_solution=True,
                                relative_tolerance=1e-12,
                                verbose=True)
    overlap = pymbar.other_estimators.bar_overlap(w_F, w_R)

    delta_f_kcal = result['Delta_f'] * kB * T * hartree_to_kcal
    delta_f_error_kcal = result['dDelta_f'] * kB * T * hartree_to_kcal

    delta_f_fwd = exp_avg_f['Delta_f'] * kB * T * hartree_to_kcal
    delta_f_bwd = exp_avg_b['Delta_f'] * kB * T * hartree_to_kcal

    return delta_f_kcal, delta_f_error_kcal, overlap, vv_energies, ss_energies, vs_energies, sv_energies, delta_f_fwd, delta_f_bwd


# Define the likelihood function using GFN-related energy calculation
def likelihood_v(angle_0, angle_1, angle_2, angle_3, angle_4, angle_5, angle_6, angle_7):
    angles = torch.tensor([angle_0, angle_1, angle_2, angle_3, angle_4, angle_5, angle_6, angle_7]).unsqueeze(0)  # Create tensor from angles

    # Example energy calculation using angles (assuming these are torsion angles)
    energy_v = v_energy.energy(angles)  # Virtual energy

    # Compute log-probabilities as -energy (since energy is proportional to -log(prob))
    log_prob = -(energy_v).item()  # Return log-probability
    return log_prob


def likelihood_s(angle_0, angle_1, angle_2, angle_3, angle_4, angle_5, angle_6, angle_7):
    angles = torch.tensor([angle_0, angle_1, angle_2, angle_3, angle_4, angle_5, angle_6, angle_7]).unsqueeze(0)  # Create tensor from angles

    # Example energy calculation using angles (assuming these are torsion angles)
    energy_s = s_energy.energy(angles)  # Virtual energy

    # Compute log-probabilities as -energy (since energy is proportional to -log(prob))
    log_prob = -(energy_s).item()  # Return log-probability
    return log_prob


def run_mcmc(likelihood, n_dims, tries=100, min_max=(-np.pi, np.pi), Rminus1_stop=0.05):

    # Cobaya config for sampling the likelihood based on torsion angles
    config = {
        "likelihood": {"gfn": likelihood},  # Likelihood function using GFN
        "sampler": {"mcmc": {"max_tries": tries}},  # Sampler using MCMC
        "params": {
            f"angle_{i}": {
                "prior": {"min": min_max[0], "max": min_max[1]},  # Range of angles
                "ref": 0,  # Starting point for each angle
                "proposal": 0.02  # Step size for angle proposals
            }
            for i in range(n_dims)  # Loop over the number of dimensions (torsion angles)
        },
        "sampler": {
            "mcmc": {
                "Rminus1_stop": Rminus1_stop, "max_tries": 1000
            }
        }
    }

    # Run the Cobaya MCMC sampler with the defined config
    updated_info, sampler = run(config)

    # Results are stored in `updated_info` and `sampler`
    print("Updated information: ", updated_info)

    samples = sampler.products()['sample'][['angle_0', 'angle_1', 'angle_2', 'angle_3', 'angle_4', 'angle_5', 'angle_6', 'angle_7']].values
    return samples, sampler

def fed_estimate_Z(energy_v, energy_s, beta_applied=False):
    kT_logZ_vacuum = calc_fed(energy_v, beta_applied=beta_applied)
    kT_logZ_solvent = calc_fed(energy_s, beta_applied=beta_applied)

    fed_Z = kT_logZ_solvent - kT_logZ_vacuum

    return fed_Z


def fed_estimate_Z_from_file(energy_v_file, energy_s_file, beta_applied=False):
    energy_v = torch.from_numpy(np.load(energy_v_file)) 
    energy_s = torch.from_numpy(np.load(energy_s_file))

    return fed_estimate_Z(energy_v, energy_s, beta_applied=beta_applied)
