import os
import csv
import sys
import torch
from openmm import unit
from tqdm import tqdm
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))
from energies import MoleculeFromSMILES_XTB
from utils import logmeanexp

# Constants
n=2
input_file = 'database.txt'
output_file = f'fed_results/rotation_fed_n={n}.csv'
batch_size = 32
kB = unit.BOLTZMANN_CONSTANT_kB.value_in_unit(unit.hartree/unit.kelvin)
beta = 1 / (kB * 298.15)
hartree_to_kcal = 627.503
T = 298.15

# Read input SMILES
molecules_smiles = {}
with open(input_file, 'r') as infile:
    reader = csv.reader(infile, delimiter=';')
    for row in reader:
        if row[0].startswith('#'):
            continue  # Skip header or comment lines
        smiles = row[1]
        experimental_val = row[3]
        experimental_uncertainty = row[4]
        molecules_smiles[smiles] = experimental_val, experimental_uncertainty

# Check for existing results and skip already processed SMILES
processed_smiles = set()
if os.path.exists(output_file):
    with open(output_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            processed_smiles.add(row[0])

results_dict = {}

# Process each SMILES
for k, v in molecules_smiles.items():
    if k in processed_smiles:
        print(f'Skipping {k} as it is already processed.')
        continue

    v_energy = MoleculeFromSMILES_XTB(k, temp=T, solvate=False)
    s_energy = MoleculeFromSMILES_XTB(k, temp=T, solvate=True)
    ground_truth = float(v[0])
    uncertainty = float(v[1])

    default_ta_vals = v_energy.rd_conf.get_freely_rotatable_tas_values()

    torsions = v_energy.tas
    if len(torsions) == 0:
        print(f'No torsions found for {k}')
        continue

    n = n if len(torsions) >= 1 else len(torsions)
    angles = torch.linspace(0, 360, 73)[:-1]
    grids = [angles] * n
    grid = torch.meshgrid(*grids, indexing='ij')
    cartesian_product = torch.stack(grid, dim=-1).reshape(-1, n)

    cartesian_product = torch.cat(
        (cartesian_product, torch.tensor(default_ta_vals[n:]).unsqueeze(0).repeat(cartesian_product.shape[0], 1)),
        dim=1
    )

    num_batches = cartesian_product.shape[0] // batch_size
    if cartesian_product.shape[0] % batch_size != 0:
        num_batches += 1

    v_energies = torch.zeros(cartesian_product.shape[0])
    s_energies = torch.zeros(cartesian_product.shape[0])
    for i in tqdm(range(num_batches)):
        start = i * batch_size
        end = min((i + 1) * batch_size, cartesian_product.shape[0])
        batch = cartesian_product[start:end]
        v_energies[start:end] = v_energy.energy(batch)
        s_energies[start:end] = s_energy.energy(batch)

    v_energies *= beta
    s_energies *= beta

    factor = hartree_to_kcal * kB * T
    v_free_energy = -logmeanexp(-v_energies) * factor
    s_free_energy = -logmeanexp(-s_energies) * factor

    fed = s_free_energy - v_free_energy
    results_dict[k] = fed, ground_truth, uncertainty
    print(f'Error for {k}: {fed - ground_truth}')

    # Write result immediately to CSV
    with open(output_file, 'a') as f:
        timestamp = datetime.now().strftime('%d-%m-%Y %H-%M')
        f.write(f'{k},{ground_truth} ± {uncertainty},0 ± 0,{fed} ± 0,0 ± 0,{timestamp}\n')

