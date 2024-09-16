import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
# from .xtb_energy import XTBEnergy, XTBBridge
from .base_set import BaseSet
from .alanine import load_data, plot_rama_traj
#from xtb.libxtb import VERBOSITY_MUTED, VERBOSITY_MINIMAL
#from xtb.interface import Calculator, Param, XTBException
#from xtb.ase.calculator import XTB
#from xtb.interface import Environment
from .utils import RDKitConformer
from .utils import torsions_to_conformations, bas_bls_to_conformations
from joblib import Parallel, delayed
from .xtbcli import get_energy
import gc
from openmm import unit

T = 298.15  # Temperature in Kelvin
beta = 1/(unit.BOLTZMANN_CONSTANT_kB.value_in_unit(unit.hartree/unit.kelvin) * T)  # Inverse temperature


class MoleculeFromSMILES_XTB(BaseSet):
    def __init__(self, smiles, temp=300, solvate=False):
        # Initialize RDKit molecule
        self.smiles = smiles
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.rd_conf = RDKitConformer(smiles, solvation=solvate)
        self.atomic_numbers = torch.tensor(self.rd_conf.get_atomic_numbers()).to(self.device)
        self.tas = self.rd_conf.freely_rotatable_tas
        self.bonds = self.rd_conf.bonds
        self.bas = self.rd_conf.bond_angles 
        if smiles == 'C[C@@H](C(=O)NC)NC(=O)C':
            self.tas = ((0, 1, 2, 3), (0, 1, 6, 7))
        self.data_ndim = len(self.tas)
        self.atom_nr = len(self.atomic_numbers)
        # Initialize XTB Energy
        self.solvate = solvate
        self.solvent = 'water' if solvate else ''
    
    def energy(self, xyz):
        # confs = bas_bls_to_conformations(xyz, self.bonds, self.bas, self.rd_conf, self.device)
        confs = torsions_to_conformations(xyz, self.tas, self.rd_conf, self.device)
        #energies = self.target.energy(confs.reshape(-1, self.atom_nr, 3)).squeeze()

        energies = []

        energies.extend(
            Parallel(n_jobs=6)(
                delayed(get_energy)(self.atomic_numbers, conf, 'gfn2', solvent=self.solvate) for conf in confs
            )
        )
        energies = torch.tensor(energies, device=self.device) * beta

        return energies
    
    def force(self, xyz):
        confs = torsions_to_conformations(xyz, self.tas, self.rd_conf, self.device)
        forces = self.target.force(confs.reshape(-1, self.atom_nr, 3)).squeeze()
        return forces

    def sample(self, batch_size):
        return None

    def time_test(self):
        import time
        
        
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

