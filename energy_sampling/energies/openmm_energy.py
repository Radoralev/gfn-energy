import torch
import bgflow as bg
from openff.toolkit import Molecule
from .base_set import BaseSet
import openmm
from openmm import app, unit, LangevinIntegrator, Vec3, Platform
from openmm.app import  Modeller, ForceField
from openmmforcefields.generators import GAFFTemplateGenerator
from tqdm import tqdm
import os 
import numpy as np
from .utils import RDKitConformer
from .utils import torsions_to_conformations

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

def create_simulation_solvent(smiles, 
                                       step_size,
                                       friction_coeff,
                                       temperature,
                                       ligand_force_field,
                                       solvate=True,
                                       use_gpu=False):
    temperature = temperature * unit.kelvin

    ligand_mol = Molecule.from_smiles(smiles)
    ligand_mol.generate_conformers(n_conformers=1)
    gaff = GAFFTemplateGenerator(molecules=ligand_mol)
    lig_top = ligand_mol.to_topology()
    print(lig_top.get_positions().reshape(-1, 3).std())    
    modeller = Modeller(lig_top.to_openmm(), lig_top.get_positions().to_openmm())

    forcefield = ForceField()
    if solvate:
        print('Adding solvent.')
        forcefield = ForceField('amber10.xml', 'implicit/gbn2.xml')
    forcefield.registerTemplateGenerator(gaff.generator)
    modeller = Modeller(ligand_mol.to_topology().to_openmm(),
                        ligand_mol.to_topology().get_positions().to_openmm())
    print('System has %d atoms before solvation' % modeller.topology.getNumAtoms())

    # if solvate:
    #     modeller.addSolvent(forcefield, model='tip3p', padding= 1.2 * unit.angstroms,
    #                         #positiveIon='Na+', negativeIon='Cl-',
    #                         ionicStrength=0 * unit.molar, neutralize=False) #boxSize=Vec3(5,5,5) * unit.nanometers)

    #     print('System has %d atoms after solvation' % modeller.topology.getNumAtoms())
    modeller.topology.setPeriodicBoxVectors(np.eye(3)*2.4)

    # Create the system
    system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.CutoffPeriodic,
        nonbondedCutoff=0.9*unit.nanometer, constraints=app.HBonds) 
    if solvate:
        system.addForce(openmm.MonteCarloBarostat(1 * unit.atmospheres, temperature, 25))
    else:
        system = forcefield.createSystem(modeller.topology)
        
    friction_coeff = friction_coeff / unit.picosecond
    step_size = step_size * unit.picoseconds
    
    if system.usesPeriodicBoundaryConditions():
        print('Default Periodic box: {}'.format(system.getDefaultPeriodicBoxVectors()))
    else:
        print('No Periodic Box')


    integrator = LangevinIntegrator(temperature, friction_coeff, step_size)

    return system, integrator, modeller.topology, Molecule.from_smiles(smiles)


class OpenMMEnergy(BaseSet):
    
    def __init__(self, smiles, temp=300, solvate=True):
        # Initialize RDKit molecule
        self.smiles = smiles
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        system, integrator, topology, self.ligand = create_simulation_solvent(smiles, 0.002, 1, temp, 'gaff-2.11', solvate=solvate)
        # Initialize RDKit molecule
        self.smiles = smiles
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.rd_conf = RDKitConformer(smiles)
        self.atomic_numbers = torch.tensor(self.rd_conf.get_atomic_numbers()).to(self.device)
        self.tas = self.rd_conf.freely_rotatable_tas
        self.bonds = self.rd_conf.bonds
        if smiles == 'C[C@@H](C(=O)NC)NC(=O)C':
            self.tas = ((0, 1, 2, 3), (0, 1, 6, 7))
        self.data_ndim = len(self.tas)
        self.atom_nr = len(self.atomic_numbers)

        openmm_bridge = bg.OpenMMBridge(system, integrator, n_workers=1)
        self.target = bg.OpenMMEnergy(dimension=self.atom_nr, bridge=openmm_bridge).to(device)
        grad_clipping = bg.utils.ClipGradient(clip=3, norm_dim=3)
        self.core_energy = bg.GradientClippedEnergy(self.target, grad_clipping).to(device)
       # self.target = bg.LinLogCutEnergy(self.target, high_energy=self.min_val*0.75+self.max_val*0.25, max_energy=self.max_val)

        
    def energy(self, xyz):
        confs = torsions_to_conformations(xyz, self.tas, self.bonds, self.rd_conf, self.device)
        energies = torch.clamp(self.target.energy(confs.reshape(-1, self.atom_nr*3)).squeeze() / 2625.5, None, 100)
        return energies

    def update_linlog(self):
        if self.min_val and self.max_val:
            self.target = bg.LinLogCutEnergy(self.core_energy, high_energy=(self.min_val*0.75+self.max_val*0.25)-self.min_val, max_energy=self.max_val)
        else:
            self.target = self.core_energy

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
