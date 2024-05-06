import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from bgflow import XTBEnergy, XTBBridge
from .base_set import BaseSet
from .alanine import load_data, plot_rama_traj
from xtb.libxtb import VERBOSITY_MUTED

class MoleculeFromSMILES(BaseSet):
    def __init__(self, smiles, temp=300, solvent="water"):
        # Initialize RDKit molecule
        self.smiles = smiles
        self.rdk_mol = Chem.MolFromSmiles(smiles)
        #if solvent == "water":
        self.rdk_mol = Chem.AddHs(self.rdk_mol)
        AllChem.EmbedMolecule(self.rdk_mol)
        self.temp = temp
        self.solvent = solvent
        
        # Extract atomic numbers
        self.atomic_numbers = np.array([atom.GetAtomicNum() for atom in self.rdk_mol.GetAtoms()])
        self.data_ndim = 3 * len(self.atomic_numbers)
        # Initialize XTB Energy
       # self.target = XTBEnergy(XTBBridge(numbers=self.atomic_numbers, temperature=temp, solvent=solvent, method='gfnff', verbosity=VERBOSITY_MUTED))
        # Get positions
        self.positions = self.rdk_mol.GetConformer().GetPositions()
    
    def energy(self, xyz):
        self.target = XTBEnergy(XTBBridge(numbers=self.atomic_numbers, temperature=self.temp, solvent=self.solvent, method='gfnff', verbosity=VERBOSITY_MUTED))
        return self.target.energy(torch.tensor(xyz.reshape(-1, len(self.atomic_numbers), 3)))
    
    def sample(self, batch_size):
        return None

# Example usage:
smiles = "CC(C)C(=O)NC(C)C(=O)NC"
molecule = MoleculeFromSMILES(smiles)
xyz = molecule.positions
energy = molecule.energy(xyz)
print(f"Energy: {energy}")
