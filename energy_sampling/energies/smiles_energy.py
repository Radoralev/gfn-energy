import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from bgflow import XTBEnergy, XTBBridge
from base_set import BaseSet

class MoleculeFromSMILES(BaseSet):
    def __init__(self, smiles, temp=300, solvent="water"):
        # Initialize RDKit molecule
        self.smiles = smiles
        self.rdk_mol = Chem.MolFromSmiles(smiles)
        if solvent == "water":
            self.rdk_mol = Chem.AddHs(self.rdk_mol)
        AllChem.EmbedMolecule(self.rdk_mol)
        
        # Extract atomic numbers
        self.atomic_numbers = np.array([atom.GetAtomicNum() for atom in self.rdk_mol.GetAtoms()])
        
        # Initialize XTB Energy
        self.target = XTBEnergy(XTBBridge(numbers=self.atomic_numbers, temperature=temp, solvent=solvent))
        
        # Get positions
        self.positions = self.rdk_mol.GetConformer().GetPositions()
    
    def energy(self, xyz):
        return self.target.energy(torch.tensor(xyz))

# Example usage:
smiles = "CC(C)C(=O)NC(C)C(=O)NC"
molecule = MoleculeFromSMILES(smiles)
xyz = molecule.positions
energy = molecule.energy(xyz)
print(f"Energy: {energy}")
