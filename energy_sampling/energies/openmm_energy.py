import torch
import bgflow as bg
from openff.toolkit import Molecule
from openmmforcefields.generators import SystemGenerator
from .base_set import BaseSet
from openmm import app, unit, LangevinIntegrator, Vec3, Platform
from openmm.app import PDBFile, Simulation, Modeller, PDBReporter, StateDataReporter, DCDReporter
from tqdm import tqdm
import os 
import numpy as np

def get_platform(use_gpu=False):
    os_platform = os.getenv('PLATFORM')
    print(os_platform)
    if os_platform:
        platform = Platform.getPlatformByName(os_platform)
    elif not use_gpu:
        return None
    else:
        # work out the fastest platform
        speed = 0
        for i in range(Platform.getNumPlatforms()):
            p = Platform.getPlatform(i)
            print(p.getName(), p.getSpeed())
            if p.getSpeed() > speed:
                platform = p
                speed = p.getSpeed()

    print('Using platform', platform.getName())

    # if it's GPU platform set the precision to mixed
    if platform.getName() == 'CUDA' or platform.getName() == 'OpenCL':
        platform.setPropertyDefaultValue('Precision', 'mixed')
        print('Set precision for platform', platform.getName(), 'to mixed')

    return platform

def create_simulation_implicit_solvent(smiles, 
                                       step_size,
                                       friction_coeff,
                                       temperature,
                                       ligand_force_field,
                                       solvate=True,
                                       use_gpu=False):
    temperature = temperature * unit.kelvin

    ligand_mol = Molecule.from_smiles(smiles)
    ligand_mol.generate_conformers(n_conformers=1)
    lig_top = ligand_mol.to_topology()
    print(lig_top.get_positions())
    modeller = Modeller(lig_top.to_openmm(), lig_top.get_positions().to_openmm())

    forcefield_kwargs = {
        'constraints': app.HBonds, 
        'rigidWater': True, 
        'removeCMMotion': False, 
        'hydrogenMass': 4*unit.amu
    }
    forcefields = []
    if solvate:
        print('Adding solvent.')
        forcefields.append('implicit/gbn2.xml')
    system_generator = SystemGenerator(
        forcefields=forcefields, 
        forcefield_kwargs=forcefield_kwargs,
        small_molecule_forcefield=ligand_force_field)

    modeller = Modeller(ligand_mol.to_topology().to_openmm(),
                        ligand_mol.to_topology().get_positions().to_openmm())

    # Create the system
    system = system_generator.create_system(modeller.topology,
                                            molecules=ligand_mol)  # Example using OBC2 model
    print(system)
    friction_coeff = friction_coeff / unit.picosecond
    step_size = step_size * unit.picoseconds

    integrator = LangevinIntegrator(temperature, friction_coeff, step_size)

    return system, integrator, modeller.topology, Molecule.from_smiles(smiles)


class OpenMMEnergy(BaseSet):
    
    def __init__(self, smiles, temp=300, solvate=True):
        # Initialize RDKit molecule
        self.smiles = smiles
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        system, integrator, topology, self.ligand = create_simulation_implicit_solvent(smiles, 0.002, 1, temp, 'gaff-2.11', solvate=solvate)
        
        n_atoms = topology.getNumAtoms()
        openmm_bridge = bg.OpenMMBridge(system, integrator, n_workers=1)
        self.data_ndim = 3 * n_atoms
        self.target = bg.OpenMMEnergy(dimension=self.data_ndim, bridge=openmm_bridge).to(device)
        print(f'System has {n_atoms} atoms')
    def energy(self, xyz):
        return torch.clamp(self.target.energy(xyz), 0, 200)
    
    def sample(self, batch_size):
        return self.ligand.generate_conformers(n_conformers=batch_size)

    def time_test(self):
        import time
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # generate random data
        time_now = time.time()
        #for i in range(25000):
        x = torch.randn(300, self.data_ndim).to(device)
        energy = self.target.energy(x)
        print('Time taken to compute energy:', time.time()-time_now)
