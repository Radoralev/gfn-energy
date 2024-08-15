import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from .xtb_energy import XTBEnergy, XTBBridge
from .base_set import BaseSet
from .alanine import load_data, plot_rama_traj
#from xtb.libxtb import VERBOSITY_MUTED, VERBOSITY_MINIMAL
#from xtb.interface import Calculator, Param, XTBException
#from xtb.ase.calculator import XTB
#from xtb.interface import Environment
from .rdkit_conformer import RDKitConformer
from .utils import torsions_to_conformations


class MoleculeFromSMILES_XTB(BaseSet):
    def __init__(self, smiles, temp=300, solvate=False):
        # Initialize RDKit molecule
        self.smiles = smiles
        self.rd_conf = RDKitConformer(smiles)
        self.atomic_numbers = torch.tensor(self.rd_conf.get_atomic_numbers()).to(self.device)
        self.tas = self.rd_conf.freely_rotatable_tas
        if smiles == 'C[C@@H](C(=O)NC)NC(=O)C':
            self.tas = ((0, 1, 2, 3), (0, 1, 6, 7))
        self.data_ndim = len(self.tas)
        self.atom_nr = len(self.atomic_numbers)
        # Initialize XTB Energy
        self.solvent = 'water' if solvate else ''
        self.target = XTBEnergy(XTBBridge(numbers=self.atomic_numbers, temperature=temp, solvent=self.solvent, method='GFN2-xTB'))#.to('cuda')        # Get positions
        self.positions = self.rdk_mol.GetConformer().GetPositions()
    
    def energy(self, xyz):
        confs = torsions_to_conformations(xyz, self.tas, self.rd_conf, self.device)
        energies = self.target.energy(confs.reshape(-1, self.atom_nr, 3)).squeeze()
        return energies
    
    def sample(self, batch_size):
        return None

    def time_test(self):
        import time
        env = Environment()
        
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # generate random data
        time_now = time.time()
        target = XTBEnergy(XTBBridge(numbers=self.atomic_numbers, temperature=self.temp, solvent=self.solvent, method='GFN-FF', verbosity=0))
        for i in range(25000):
            for i in range(300):
                x = torch.randn(1, int(self.data_ndim/3), 3).to(device)
                #calc.set_verbosity(VERBOSITY_MINIMAL)
                energy = target.energy(x)
        print('Time taken to compute energy:', time.time()-time_now)

