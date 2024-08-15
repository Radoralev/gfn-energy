import torch
import numpy as np
import bgflow as bg
import mdtraj as md
import openmm
import bgmol
from torch.utils.data import Dataset
from torch.distributions import MixtureSameFamily, Normal, Categorical
from .base_set import BaseSet
from openmm import unit
from bgflow.utils.types import assert_numpy
from matplotlib.colors import LogNorm
from matplotlib import pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
from bgflow import XTBEnergy, XTBBridge
import time
import torchani
from energies.neural_energy import NeuralEnergy

def plot_rama_traj(trajectory, w=None, get_phi=False, i=-1, model=None):
    def get_phi_psi(trajectory, i=-1, model=None):
        if not isinstance(trajectory, md.Trajectory):
            if isinstance(trajectory, torch.Tensor):
                trajectory = assert_numpy(trajectory.view(len(trajectory), *model.positions.shape))
            trajectory = md.Trajectory(trajectory, model.mdtraj_topology)
        phi = md.compute_phi(trajectory)[1][:,i]
        psi = md.compute_psi(trajectory)[1][:,i]
        return phi, psi
    
    phi, psi = get_phi_psi(trajectory, i, model)
    plot_range = [-np.pi, np.pi]

    # Create figure and axis using subplots
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    
    # Plot histogram
    ax.hist2d(phi, psi, 60, weights=w, norm=LogNorm(), range=[plot_range, plot_range])
    ax.set_xlim(plot_range)
    ax.set_ylim(plot_range)
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$\psi$")
    ax.set_box_aspect(1)
    

    return fig, ax

def get_phi(trajectory, model, i=-1):
    if not isinstance(trajectory, md.Trajectory):
        trajectory = md.Trajectory(trajectory, model.mdtraj_topology)
    phi = md.compute_phi(trajectory)[1][:,i]
    return phi

def split_by_phi(data, model):
    phis = get_phi(data, model)
    indices = np.logical_or(phis < 0, 2.15 < phis) # source data
    return data[indices], data[np.logical_not(indices)]

def load_data(temp, model, device, phi):
        ctx = {"device": device, "dtype": torch.float32}
        filename = f'data/alanine/MD-AlanineDipeptideVacuum-T{temp}.npz'
        MDdata = np.load(filename)['data']
        # MDener = np.load(filename)['ener']

        source_data, target_data = split_by_phi(MDdata, model)
        if phi == 'full':
            return torch.tensor(MDdata).view(-1, 3*len(model.positions)).to(**ctx)
        elif phi == 'source':
            return torch.tensor(source_data).view(-1, 3*len(model.positions)).to(**ctx)
        else:
            return torch.tensor(target_data).view(-1, 3*len(model.positions)).to(**ctx)


def get_bgmol_model(system_name, temperature=None):
        
        model = bgmol.system_by_name(system_name.replace('ModifiedPSI',''))
        
        if 'ModifiedPSI' in system_name:
            extraBias_str = '100*sin(0.5*theta)^2'
            extraBias = openmm.CustomTorsionForce(extraBias_str)
            psi_angles = md.compute_psi(md.Trajectory(model.positions, model.mdtraj_topology))[0]
            for i in range(len(psi_angles)):
                extraBias.addTorsion(*psi_angles[i])
                print(f"{system_name}, adding bias on psi{psi_angles[i]}: {extraBias_str}")
            model.system.addForce(extraBias)        
        
        if temperature is not None:
            model.reinitialize_energy_model(temperature=temperature)
        
        return model

class Alanine(BaseSet):
    def __init__(self, device, phi='source', len_data=-2333, temp=300, system_name="AlanineDipeptideVacuum", energy=None):
        super().__init__(len_data)
        self.device = device
        self.temp = temp        
        self.phi = phi
        self.min_val = None
        self.bgmol_model = get_bgmol_model(system_name, temperature=temp)
        self.data = load_data(temp, self.bgmol_model, device, phi)
        self.smiles = 'C[C@@H](C(=O)NC)NC(=O)C'
        self.energy_model = energy
        if self.energy_model:
            self.data_ndim = self.energy_model.data_ndim

    def energy(self, x):    
        energies = self.energy_model.energy(x).squeeze()
        return energies

    def sample(self, batch_size):
        indices = np.random.choice(len(self.data), size=batch_size, replace=False)
        return self.data[indices] 

    def plot(self, x, w=None, sampled=False):
        if not sampled:
            x = self.energy_model.torsions_to_conformations(x)
        return plot_rama_traj(x, w, get_phi=True, model=self.bgmol_model)
    
    def time_test(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # generate random data
        x = torch.randn(1024, self.data_ndim).to(device)
        time_now = time.time()
        energy = self.energy(x)
        print('Time taken to compute energy:', time.time()-time_now)
