import torch
from .base_set import BaseSet
from models.utils import smiles2graph, prep_input
from torch_geometric import loader
from torch_geometric.data import Batch


class NeuralEnergy(BaseSet):
    def __init__(self, model, smiles, batch_size_train, batch_size_val=512, batch_size_final_val=2048):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.model.eval()
        self.graph = smiles2graph(smiles)
        self.data_ndim = 3 * self.graph['num_nodes']
        self.atoms = torch.from_numpy(self.graph['node_feat'])
        self.batch_size_final_val = batch_size_final_val
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val
        self.min_val = None
        self.max_val = None

        data_list = prep_input(self.graph, pos=torch.ones(batch_size_train, self.data_ndim//3, 3) ,device=self.device)
        self.batch_train = Batch.from_data_list(data_list)
        data_list = prep_input(self.graph, pos=torch.ones(batch_size_val, self.data_ndim//3, 3) ,device=self.device)
        self.batch_val = Batch.from_data_list(data_list)
        data_list = prep_input(self.graph, pos=torch.ones(batch_size_final_val, self.data_ndim//3, 3) ,device=self.device)
        self.batch_final_val = Batch.from_data_list(data_list)

    def energy(self, xyz):
        if xyz.shape[0] == self.batch_size_train:
            batch = self.batch_train
        elif xyz.shape[0] == self.batch_size_val:
            batch = self.batch_val
        elif xyz.shape[0] == self.batch_size_final_val:
            batch = self.batch_final_val
        
        batch.pos = xyz.reshape(-1, 3)
        energies = self.model(batch).squeeze()
        return energies
    
    def sample(self, batch_size):
        return None
    

class SolvationEnergy(BaseSet):
    def __init__(self, energy_solv: NeuralEnergy=None, energy_vac: NeuralEnergy=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.energy_solvent = energy_solv.to(self.device)
        self.energy_vacuum = energy_vac.to(self.device)
        

    def energy(self, data):
        xyz, solv_flag = data
        # the solvation flag is a boolean tensor (bs,), this function applies 
        # the solvent energy to the samples where the flag is True and the vacuum energy to the rest
        solv_idx = solv_flag.nonzero().squeeze()
        vac_idx = (~solv_flag).nonzero().squeeze()
        solv_energies = self.energy_solvent.energy(xyz[solv_idx])
        vac_energies = self.energy_vacuum.energy(xyz[vac_idx])
        energies = torch.zeros_like(solv_flag, dtype=torch.float32)
        energies[solv_idx] = solv_energies
        energies[vac_idx] = vac_energies
        return energies
    
    def sample(self, batch_size):
        return None
