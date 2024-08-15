import torch
import torchani
import os
import math
import torch.utils.tensorboard
import tqdm
import pickle

# helper function to convert energy unit from Hartree to kcal/mol
from torchani.units import hartree2kcalmol

# device to run the training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')





class TorchANI_Local(torch.nn.Module):
    def __init__(self, ):
        super(TorchANI_Local, self).__init__()

        self.ANI2x = torchani.models.ANI2x(periodic_table_index=True).to('cuda')

        species_order = self.ANI2x.species
        num_species = len(species_order)
        aev_computer = self.ANI2x.aev_computer#torchani.AEVComputer(Rcr, Rca, EtaR, ShfR, EtaA, Zeta, ShfA, ShfZ, num_species)
        energy_shifter = self.ANI2x.energy_shifter


        self.nn = self.ANI2x.neural_networks.to(device)
        self.model = torchani.nn.Sequential(aev_computer, self.nn).to(device)

    def forward(self, x):
        x = self.ANI2x.species_converter(x)
        return self.model(x)
    
    def load(self, path):
            checkpoint = torch.load(path)
            self.nn.load_state_dict(checkpoint)
