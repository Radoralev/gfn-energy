'''
File to define Neural Networks
'''

# import torch_cluster
from torch_geometric.nn import radius_graph
from torch.nn import PairwiseDistance
import torch
from torch import nn
from torch_scatter import scatter
from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict, List
from torch.cuda.amp import autocast

T = TypeVar('T', bound='Module')

from .GNN_Layers import GBNeck_interaction, GBNeck_energies, IN_layer_all_swish_2pass

torch.backends.cudnn.benchmark = True

class GNN_GBNeck(torch.nn.Module):
    def __init__(self, radius=0.4, max_num_neighbors=32, parameters=None, device=None, jittable=False,unique_radii=None):
        '''
        GNN to reproduce the GBNeck Model
        '''
        super().__init__()

        # In order to be differentiable all tensors *need* to be created on the same device
        if device is None:
            self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self._device = device

        self._nobatch = False

        # Initiate Graph Builder
        if not parameters is None:
            self._gbparameters = torch.tensor(parameters,dtype=torch.float,device=self._device)
        self.r = radius
        self._max_num_neighbors = max_num_neighbors
        #self._grapher = RadiusGraph(r=self._radius, loop=False, max_num_neighbors=self._max_num_neighbors)

        # Init Distance Calculation
        self._distancer = PairwiseDistance()
        self._jittable = jittable
        if jittable:
            self.aggregate_information = GBNeck_interaction(parameters,self._device,unique_radii=unique_radii).jittable()
            self.calculate_energies = GBNeck_energies(parameters,self._device,unique_radii=unique_radii).jittable()
        else:
            self.aggregate_information = GBNeck_interaction(parameters,self._device,unique_radii=unique_radii)
            self.calculate_energies = GBNeck_energies(parameters,self._device,unique_radii=unique_radii)

        self.lin = nn.Linear(1,1)

    def get_edge_features(self, distances, alpha=2, max_range=0.4, min_range=0.1, num_kernels=32):
        m = alpha * (max_range - min_range) / (num_kernels + 1)
        lower_bound = min_range + m
        upper_bound = max_range - m
        centers = torch.linspace(lower_bound, upper_bound, num_kernels, device=self._device)
        k = distances - centers
        return torch.maximum(torch.tensor(0, device=self._device), torch.pow(1 - (k / m) ** 2, 3))

    def build_graph(self, data):

        # Get Radius Graph
        #graph = self._grapher(data)

        # Extract edge index
        edge_index = radius_graph(
            data.pos,
            self.r,
            data.batch,
            False,
            max_num_neighbors=self.max_num_neighbors,
        )


        # Extract node features
        node_features = data.atoms

        # Extract edge features
        distances = self._distancer(data.pos[edge_index[0]], data.pos[edge_index[1]])

        # For GBNeck model distances are features
        edge_attributes = distances.unsqueeze(1)

        return node_features, edge_index, edge_attributes

    def forward(self, data):

        # Enable tracking of gradients
        # Get input as Tensor create on device
        data.pos = data.pos.clone().detach().requires_grad_(True)
        # Build Graph
        _, edge_index, edge_attributes = self.build_graph(data)

        x = self._gbparameters.repeat(torch.max(data.batch)+1,1)

        # Do message passing
        Bc = self.aggregate_information(x=x, edge_index=edge_index, edge_attributes=edge_attributes) # B and charges
        energies = self.calculate_energies(x=Bc, edge_index=edge_index, edge_attributes=edge_attributes)

        # Return prediction and Gradients with respect to data
        gradients = torch.autograd.grad(energies.sum(), inputs=data.pos, create_graph=True)[0]
        forces = -1 * gradients

        if self._nobatch:
            energy = energies.sum()
            energy = energy.unsqueeze(0)
            energy = energy.unsqueeze(0)
        else:
            energy = torch.empty((torch.max(data.batch) + 1,1), device=self._device)
            for batch in data.batch.unique():
                energy[batch] = energies[torch.where(data.batch == batch)].sum()

        return energy, forces

class GNN_GBNeck_2(GNN_GBNeck):

    def forward(self, data):

        # Enable tracking of gradients
        # Get input as Tensor create on device
        data.pos = data.pos.clone().detach().requires_grad_(True)

        # Build Graph
        _, edge_index, edge_attributes = self.build_graph(data)

        x = data.atom_features

        # Do message passing
        Bc = self.aggregate_information(x=x, edge_index=edge_index, edge_attributes=edge_attributes) # B and charges
        energies = self.calculate_energies(x=Bc, edge_index=edge_index, edge_attributes=edge_attributes)

        # Return prediction and Gradients with respect to data
        gradients = torch.autograd.grad(energies.sum(), inputs=data.pos, create_graph=True)[0]
        forces = -1 * gradients

        if self._nobatch:
            energy = energies.sum()
            energy = energy.unsqueeze(0)
            energy = energy.unsqueeze(0)
        else:
            energy = torch.empty((torch.max(data.batch) + 1,1), device=self._device)
            for batch in data.batch.unique():
                energy[batch] = energies[torch.where(data.batch == batch)].sum()

        return energy, forces

    def build_graph(self, data):

        # Get Radius Graph
        #graph = self._grapher(data)

        # Extract edge index
        edge_index = radius_graph(
            data.pos,
            self.r,
            data.batch,
            False,
            max_num_neighbors=self.max_num_neighbors,
        )


        # Extract node features
        node_features = data.atoms

        # Extract edge features
        distances = self._distancer(data.pos[edge_index[0]], data.pos[edge_index[1]])

        # For GBNeck model distances are features
        edge_attributes = distances.unsqueeze(1)

        return node_features, edge_index, edge_attributes

class GNN_Grapher:

    def __init__(self,radius,max_num_neighbors) -> None:
        self.r = radius
        self.max_num_neighbors = max_num_neighbors
    def build_gnn_graph(self, data):

        # Get Radius Graph
        # graph = self._gnn_grapher(data)

        # Extract edge index
        edge_index = radius_graph(
            data.pos,
            self.r,
            data.batch,
            max_num_neighbors=self.max_num_neighbors,
            num_workers=self.num_workers,
        )


        # Extract node features
        node_features = data.atoms

        # Extract edge features
        distances = self._distancer(data.pos[edge_index[0]], data.pos[edge_index[1]])

        # For GBNeck model distances are features
        edge_attributes = distances.unsqueeze(1)

        return node_features, edge_index, edge_attributes

class GNN_Grapher_2(GNN_Grapher):

    def build_gnn_graph(self, data):

        # Get Radius Graph
        #graph = self._gnn_grapher(data)

        # Extract edge index
        edge_index = data.edge_index

        # Extract node features
        node_features = data.atoms

        # Extract edge features
        distances = self._distancer(data.pos[edge_index[0]], data.pos[edge_index[1]])

        # For GBNeck model distances are features
        edge_attributes = distances.unsqueeze(1)

        return node_features, edge_index, edge_attributes

class GNN3_all_swish_multiple_peptides_GBNeck_trainable_dif_graphs_corr_with_separate_SA(GNN_GBNeck_2,GNN_Grapher_2):

    def __init__(self,fraction=0.5,radius=0.4, max_num_neighbors=32, parameters=None, device=None, jittable=False,unique_radii=None, hidden=128):

        gbneck_radius = 10.0
        self._gnn_radius = radius
        GNN_GBNeck_2.__init__(self,radius=gbneck_radius, max_num_neighbors=max_num_neighbors, parameters=parameters, device=device, jittable=jittable,unique_radii=unique_radii)
        GNN_Grapher_2.__init__(self,radius=radius, max_num_neighbors=max_num_neighbors)

        self._fraction = fraction
        if self._jittable:
            self.interaction1 = IN_layer_all_swish_2pass(3 + 3, hidden,radius,device,hidden).jittable()
            self.interaction2 = IN_layer_all_swish_2pass(hidden + hidden, hidden,radius,device,hidden).jittable()
            self.interaction3 = IN_layer_all_swish_2pass(hidden + hidden, 2,radius,device,hidden).jittable()
        else:
            self.interaction1 = IN_layer_all_swish_2pass(3 + 3, hidden,radius,device,hidden)
            self.interaction2 = IN_layer_all_swish_2pass(hidden + hidden, hidden,radius,device,hidden)
            self.interaction3 = IN_layer_all_swish_2pass(hidden + hidden, 2,radius,device,hidden)

        self._silu = torch.nn.SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        data.pos = data.pos.clone().detach().requires_grad_(True)
        # Build Graph
        _, edge_index, edge_attributes = self.build_graph(data)
        _, gnn_edge_index, gnn_edge_attributes = self.build_gnn_graph(data)
        x = data.atoms

        # Do message passing
        Bc = self.aggregate_information(x=x, edge_index=edge_index, edge_attributes=edge_attributes)  # B and charges

        # ADD small correction
        Bcn = torch.concat((Bc,x[:,1].unsqueeze(1)),dim=1)
        Bcn = self.interaction1(edge_index=gnn_edge_index,x=Bcn,edge_attributes=gnn_edge_attributes)
        Bcn = self._silu(Bcn)
        Bcn = self.interaction2(edge_index=gnn_edge_index,x=Bcn,edge_attributes=gnn_edge_attributes)
        Bcn = self._silu(Bcn)
        Bcn = self.interaction3(edge_index=gnn_edge_index,x=Bcn,edge_attributes=gnn_edge_attributes)
        
        # Separate into polar and non-polar contributions
        c_scale = Bcn[:,0]
        sa_scale = Bcn[:,1]

        # Calculate SA term
        gamma = 0.00542 # kcal/(mol A^2)
        offset = 0.0195141
        radius = (x[:,1] + offset).unsqueeze(1)
        sasa = self.sigmoid(sa_scale.unsqueeze(1)) * (radius+0.14)**2
        sa_energies = 4.184 * gamma * sasa * 100

        # Scale the GBNeck born radii with plus minus 50%
        Bcn = Bc[:,0].unsqueeze(1) * (self._fraction + self.sigmoid(c_scale.unsqueeze(1))*(1-self._fraction)*2)

        # get 'Born' radius with charge
        Bc = torch.concat((Bcn,Bc[:,1].unsqueeze(1)),dim=1)

        # Evaluate GB energies
        energies = self.calculate_energies(x=Bc, edge_index=edge_index, edge_attributes=edge_attributes)

        # Add SA term
        energies = energies + sa_energies

        # Return prediction and Gradients with respect to data
        gradients = torch.autograd.grad(energies.sum(), inputs=data.pos, create_graph=True)[0]
        forces = -1 * gradients
        if self._nobatch:
            energy = energies.sum()
            energy = energy.unsqueeze(0)
            energy = energy.unsqueeze(0)
        else:
            energy = torch.empty((torch.max(data.batch) + 1, 1), device=self._device)
            for batch in data.batch.unique():
                energy[batch] = energies[torch.where(data.batch == batch)].sum()

        return energy, forces


class GNN3_scale_64(GNN3_all_swish_multiple_peptides_GBNeck_trainable_dif_graphs_corr_with_separate_SA):

    def __init__(self, fraction=0.5, radius=0.4, max_num_neighbors=32, parameters=None, device=None, jittable=False, unique_radii=None, hidden=64):
        super().__init__(fraction, radius, max_num_neighbors, parameters, device, jittable, unique_radii, hidden)
