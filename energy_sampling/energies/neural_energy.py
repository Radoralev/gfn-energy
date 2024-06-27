import torch
from .base_set import BaseSet
from models.utils import smiles2graph, prep_input
from torch_geometric import loader
from torch_geometric.data import Batch


class NeuralEnergy(BaseSet):
    def __init__(self, model, smiles, batch_size=1):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.model.eval()
        self.batch_size = batch_size
        self.graph = smiles2graph(smiles)
        self.data_ndim = 3 * self.graph['num_nodes']


    def energy(self, xyz):
        data_list = prep_input(self.graph, xyz.reshape(-1, self.data_ndim//3, 3), device=self.device)
        batch = Batch.from_data_list(data_list)
        energies = self.model(batch).squeeze()
        return energies
    
    def sample(self, batch_size):
        return None