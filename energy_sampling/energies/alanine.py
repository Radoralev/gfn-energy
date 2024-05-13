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
    def __init__(self, device, phi='source', len_data=-2333, temp=300, system_name="AlanineDipeptideVacuum"):
        super().__init__(len_data)
        self.device = device
        self.temp = temp        
        self.phi = phi
        self.bgmol_model = get_bgmol_model(system_name, temperature=temp)
        #bridge = bg.OpenMMBridge(self.bgmol_model.system, 
        #                              openmm.LangevinIntegrator(temp*unit.kelvin, 1/unit.picosecond, 0.002*unit.picoseconds), 
        #                              n_workers=1)
        #self.data_ndim = 66
        #self._energy = bg.OpenMMEnergy(dimension=self.data_ndim, bridge=bridge).to(device)
        self.data = load_data(temp, self.bgmol_model, device, phi)
        
        # Extract atomic numbers
        self.atomic_numbers = torch.tensor([1, 0, 0, 0, 1, 3, 2, 0, 1, 0, 1, 0, 0, 0, 1, 3, 2, 0, 1, 0, 0, 0,]).to(self.device)

        print(len(self.atomic_numbers))
        self.data_ndim = 3 * len(self.atomic_numbers)

        self._energy = torchani.models.ANI1x(periodic_table_index=True).to(self.device)
        #self._energy = XTBEnergy(XTBBridge(numbers=self.atomic_numbers, temperature=temp, solvent='', method='gfnff')).to(device)
        #time_now = time.time()
        #self.energy_cap = self._energy.energy(self.data.reshape(-1, len(self.atomic_numbers), 3)).max()
        #print('Time taken to compute energy cap:', time.time()-time_now)
        
    def energy(self, x):    
        #print(x)
        an_bs = self.atomic_numbers.unsqueeze(0).repeat(x.size(0), 1).to(self.device)
        energies = torch.clamp(self._energy((an_bs, x.reshape(-1, int(self.data_ndim/3), 3))).energies, 0, None)
        return energies
    
        energy = self._energy.energy(x.reshape(-1, len(self.atomic_numbers), 3))
        energy = torch.clamp(energy, 0, 500) #/ 10000
        if self.phi != 'full':
            x_spatial = x.view(-1, len(self.bgmol_model.positions), 3).clone().detach().cpu().numpy()
            phi = get_phi(x_spatial, self.bgmol_model)
            if self.phi == 'source':
                valid = np.logical_and(0.0 < phi, phi < 2.15)
                energy[np.logical_not(valid)] = torch.tensor(float('inf'))
            elif self.phi == 'target':
                valid = np.logical_or(phi < 0.0, 2.15 < phi)
                energy[np.logical_not(valid)] = torch.tensor(float('inf'))
        return energy

    def sample(self, batch_size):
        indices = np.random.choice(len(self.data), size=batch_size, replace=False)
        return self.data[indices] 

    def plot(self, x, w=None):
        return plot_rama_traj(x, w, get_phi=True, model=self.bgmol_model)
    
    def time_test(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # generate random data
        x = torch.randn(1024, self.data_ndim).to(device)
        time_now = time.time()
        energy = self.energy(x)
        print('Time taken to compute energy:', time.time()-time_now)
