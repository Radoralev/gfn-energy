import torch
import torchani
from rdkit import Chem
from rdkit.Chem import AllChem
from openmm import app, unit, LangevinIntegrator, Vec3, Platform
from openff.toolkit import Molecule
from openmm.app import Modeller, ForceField
from openmmforcefields.generators import GAFFTemplateGenerator
from .base_set import BaseSet
import numpy as np

def get_atom_types_from_smiles(smiles, solvate=True):
    ligand_mol = Molecule.from_smiles(smiles)
    ligand_mol.generate_conformers(n_conformers=1)
    gaff = GAFFTemplateGenerator(molecules=ligand_mol)
    lig_top = ligand_mol.to_topology()
    print(lig_top.get_positions())
    modeller = Modeller(lig_top.to_openmm(), lig_top.get_positions().to_openmm())

    forcefield = ForceField()
    if solvate:
        print('Adding solvent.')
        forcefield = ForceField('amber10.xml', 'amber14/tip3pfb.xml')
    forcefield.registerTemplateGenerator(gaff.generator)
    modeller = Modeller(ligand_mol.to_topology().to_openmm(),
                        ligand_mol.to_topology().get_positions().to_openmm())
    print('System has %d atoms before solvation' % modeller.topology.getNumAtoms())

    if solvate:
        modeller.addSolvent(forcefield, model='tip3p', padding= 1.2 * unit.angstroms,
                            #positiveIon='Na+', negativeIon='Cl-',
                            ionicStrength=0 * unit.molar, neutralize=False) #boxSize=Vec3(5,5,5) * unit.nanometers)

        print('System has %d atoms after solvation' % modeller.topology.getNumAtoms())
    modeller.topology.setPeriodicBoxVectors(np.eye(3)*2.4)
    topology = modeller.topology 
    atom_types = torch.tensor([atom.element.atomic_number for atom in topology.atoms()])
    return atom_types, modeller.getPositions()
    

class TorchANIEnergy(BaseSet):
    def __init__(self, model, smiles, batch_size=1, solvate=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.model.eval()
        self.atomic_numbers, init_pos = get_atom_types_from_smiles(smiles, solvate)
        self.atomic_numbers = self.atomic_numbers.to(self.device)
        self.data_ndim = 3 * len(self.atomic_numbers)
        self.batch_size = batch_size
        init_pos = torch.tensor(init_pos.value_in_unit(unit.angstrom)).to(self.device, dtype=torch.float32)
        print(init_pos.shape)
        print(self.atomic_numbers.shape)
        print('Atomic numbers:', self.atomic_numbers)
        print('Energy of initial conformation:', self.energy(init_pos.repeat(self.batch_size, 1, 1).to(self.device)))
        
    def energy(self, xyz):
        an_bs = self.atomic_numbers.unsqueeze(0).repeat(xyz.size(0), 1).to(self.device)
        energies = torch.clamp(self.model((an_bs, xyz.reshape(-1, int(self.data_ndim/3), 3))).energies, 0, None)
        return energies
    
    def sample(self, batch_size):
        return None
    
    def time_test(self):
        import time
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        time_now = time.time()
        for i in range(25000):
            for i in range(300):
                x = torch.randn(1, int(self.data_ndim/3), 3).to(device)
                energy = self.model((self.atomic_numbers.unsqueeze(0).repeat(x.size(0), 1), x)).energies
        print('Time taken to compute energy:', time.time()-time_now)
