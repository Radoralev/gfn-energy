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

from .utils import embed_mol_and_get_conformer, RDKitConformer, torsions_to_conformations


def get_atom_types_from_smiles(smiles, solvate=True):
    ligand_mol = Molecule.from_smiles(smiles)
    ligand_mol.generate_conformers(n_conformers=1)
    gaff = GAFFTemplateGenerator(molecules=ligand_mol)
    lig_top = ligand_mol.to_topology()
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
    return atom_types#, modeller.getPositions()
    

class TorchANIEnergy(BaseSet):
    def __init__(self, model, smiles, batch_size=1, solvate=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.model.eval()
        self.rd_conf = RDKitConformer(smiles)
        self.atomic_numbers = torch.tensor(self.rd_conf.get_atomic_numbers()).to(self.device)
        self.tas = self.rd_conf.freely_rotatable_tas
        if smiles == 'C[C@@H](C(=O)NC)NC(=O)C':
            self.tas = ((0, 1, 2, 3), (0, 1, 6, 7))
        self.data_ndim = len(self.tas)
        self.atom_nr = len(self.atomic_numbers)
        self.batch_size = batch_size

        print('Number of atoms:', self.atom_nr, ', Number of torsion angles:', self.data_ndim)
        print('Torsion angles:', self.tas)

        
    def energy(self, xyz):
        an_bs = self.atomic_numbers.unsqueeze(0).repeat(xyz.size(0), 1).to(self.device)
        confs = torsions_to_conformations(xyz, self.tas, self.rd_conf, self.device)
        energies = self.model((an_bs, confs.reshape(-1, self.atom_nr, 3))).energies
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
