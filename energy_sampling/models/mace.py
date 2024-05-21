from typing import Optional

import torch
from torch.nn import functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool
import e3nn

from models.mace_modules.irreps_tools import reshape_irreps
from models.mace_modules.blocks import (
    EquivariantProductBasisBlock,
    RadialEmbeddingBlock,
)
from models.layers.tfn_layer import TensorProductConvLayer
from torch_geometric.data import Data, Dataset, DataLoader, Batch
from ogb.utils.features import (allowable_features, atom_to_feature_vector,
 bond_to_feature_vector, atom_feature_vector_to_dict, bond_feature_vector_to_dict) 
from rdkit import Chem
import numpy as np

def smiles2graph(smiles_string):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """

    mol = Chem.MolFromSmiles(smiles_string)
    mol = Chem.AddHs(mol)
    #print(mol)
    # atoms
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(atom_to_feature_vector(atom))
    x = np.array(atom_features_list, dtype = np.int64)

    # bonds
    num_bond_features = 3  # bond type, bond stereo, is_conjugated
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = bond_to_feature_vector(bond)

            # add edges in both directions
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = np.array(edges_list, dtype = np.int64).T

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype = np.int64)

    else:   # mol has no bonds
        edge_index = np.empty((2, 0), dtype = np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype = np.int64)

    graph = dict()
    graph['edge_index'] = edge_index
    graph['edge_feat'] = edge_attr
    graph['node_feat'] = x
    graph['num_nodes'] = len(x)

    return graph 

def prep_input(graph, pos=None, device=None):
    datalist = []
    for xyz in pos:
        if pos is not None:
            graph['pos'] = xyz  
        #print('Number of atoms:', graph['node_feat'].shape[0])
        data = Data(
            atoms=torch.from_numpy(graph['node_feat'][:, 0]), 
            edge_index=torch.from_numpy(graph['edge_index']), 
            edge_attr=torch.from_numpy(graph['edge_feat']), 
            pos=graph['pos'],).to(device)
        data.validate(raise_on_error=True)
        datalist.append(data)
    batch = Batch.from_data_list(datalist).to(device)
    return batch

class MACEModel(torch.nn.Module):
    """
    MACE model from "MACE: Higher Order Equivariant Message Passing Neural Networks".
    """
    def __init__(
        self,
        r_max: float = 10.0,
        num_bessel: int = 8,
        num_polynomial_cutoff: int = 5,
        max_ell: int = 2,
        correlation: int = 3,
        num_layers: int = 5,
        emb_dim: int = 64,
        hidden_irreps: Optional[e3nn.o3.Irreps] = None,
        mlp_dim: int = 256,
        in_dim: int = 1,
        out_dim: int = 1,
        aggr: str = "sum",
        pool: str = "sum",
        batch_norm: bool = True,
        residual: bool = True,
        equivariant_pred: bool = False,
        smiles: str = None,
    ):
        """
        Parameters:
        - r_max (float): Maximum distance for Bessel basis functions (default: 10.0)
        - num_bessel (int): Number of Bessel basis functions (default: 8)
        - num_polynomial_cutoff (int): Number of polynomial cutoff basis functions (default: 5)
        - max_ell (int): Maximum degree of spherical harmonics basis functions (default: 2)
        - correlation (int): Local correlation order = body order - 1 (default: 3)
        - num_layers (int): Number of layers in the model (default: 5)
        - emb_dim (int): Scalar feature embedding dimension (default: 64)
        - hidden_irreps (Optional[e3nn.o3.Irreps]): Hidden irreps (default: None)
        - mlp_dim (int): Dimension of MLP for computing tensor product weights (default: 256)
        - in_dim (int): Input dimension of the model (default: 1)
        - out_dim (int): Output dimension of the model (default: 1)
        - aggr (str): Aggregation method to be used (default: "sum")
        - pool (str): Global pooling method to be used (default: "sum")
        - batch_norm (bool): Whether to use batch normalization (default: True)
        - residual (bool): Whether to use residual connections (default: True)
        - equivariant_pred (bool): Whether it is an equivariant prediction task (default: False)

        Note:
        - If `hidden_irreps` is None, the irreps for the intermediate features are computed 
          using `emb_dim` and `max_ell`.
        - The `equivariant_pred` parameter determines whether it is an equivariant prediction task.
          If set to True, equivariant prediction will be performed.
        """
        super().__init__()
        
        self.r_max = r_max
        self.max_ell = max_ell
        self.num_layers = num_layers
        self.emb_dim = emb_dim
        self.mlp_dim = mlp_dim
        self.residual = residual
        self.batch_norm = batch_norm
        self.hidden_irreps = hidden_irreps
        self.equivariant_pred = equivariant_pred
        self.smiles = smiles
        self.smiles_graph = smiles2graph(smiles)
        self.in_dim = in_dim
        # Edge embedding
        self.radial_embedding = RadialEmbeddingBlock(
            r_max=r_max,
            num_bessel=num_bessel,
            num_polynomial_cutoff=num_polynomial_cutoff,
        )
        sh_irreps = e3nn.o3.Irreps.spherical_harmonics(max_ell)
        self.spherical_harmonics = e3nn.o3.SphericalHarmonics(
            sh_irreps, normalize=True, normalization="component"
        )

        # Embedding lookup for initial node features
        self.emb_in = torch.nn.Embedding(in_dim, emb_dim)

        # Set hidden irreps if none are provided
        if hidden_irreps is None:
            hidden_irreps = (sh_irreps * emb_dim).sort()[0].simplify()
            # Note: This defaults to O(3) equivariant layers
            # It is possible to use SO(3) equivariance by passing the appropriate irreps

        self.convs = torch.nn.ModuleList()
        self.prods = torch.nn.ModuleList()
        self.reshapes = torch.nn.ModuleList()
        
        # First layer: scalar only -> tensor
        self.convs.append(
            TensorProductConvLayer(
                in_irreps=e3nn.o3.Irreps(f'{2*emb_dim}x0e'),
                out_irreps=hidden_irreps,
                sh_irreps=sh_irreps,
                edge_feats_dim=self.radial_embedding.out_dim,
                mlp_dim=mlp_dim,
                aggr=aggr,
                batch_norm=batch_norm,
                gate=False,
            )
        )
        self.reshapes.append(reshape_irreps(hidden_irreps))
        self.prods.append(
            EquivariantProductBasisBlock(
                node_feats_irreps=hidden_irreps,
                target_irreps=hidden_irreps,
                correlation=correlation,
                element_dependent=False,
                num_elements=in_dim,
                use_sc=residual
            )
        )

        # Intermediate layers: tensor -> tensor
        for _ in range(num_layers - 1):
            self.convs.append(
                TensorProductConvLayer(
                    in_irreps=hidden_irreps,
                    out_irreps=hidden_irreps,
                    sh_irreps=sh_irreps,
                    edge_feats_dim=self.radial_embedding.out_dim,
                    mlp_dim=mlp_dim,
                    aggr=aggr,
                    batch_norm=batch_norm,
                    gate=False,
                )
            )
            self.reshapes.append(reshape_irreps(hidden_irreps))
            self.prods.append(
                EquivariantProductBasisBlock(
                    node_feats_irreps=hidden_irreps,
                    target_irreps=hidden_irreps,
                    correlation=correlation,
                    element_dependent=False,
                    num_elements=in_dim,
                    use_sc=residual
                )
            )

        # Global pooling/readout function
        self.pool = {"mean": global_mean_pool, "sum": global_add_pool}[pool]

        if self.equivariant_pred:
            # Linear predictor for equivariant tasks using geometric features
            self.pred = torch.nn.Linear(hidden_irreps.dim, out_dim)
        else:
            # MLP predictor for invariant tasks using only scalar features
            self.pred = torch.nn.Sequential(
                torch.nn.Linear(emb_dim, emb_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(emb_dim, out_dim)
            )
    
    def forward(self, x):
        #print(x.shape)
        pos, t = x[:, :self.in_dim], x[:, self.in_dim:]
        pos = pos.reshape(-1, int(self.in_dim / 3), 3)
        bs, atom_num, _ = pos.shape
        batch = prep_input(self.smiles_graph, pos, device=self.emb_in.weight.device)
        # Node embedding
        #atom_number = batch.num_nodes
        h = self.emb_in(batch.atoms)  # (n,) -> (n, d)
        h = h.view(bs, atom_num, self.emb_dim)
        t = t.unsqueeze(1).expand(h.size(0), h.size(1), h.size(2))
        #print(h.shape)
        h = torch.cat([h, t], dim=-1)
        #print(h.shape)
        h = h.view(bs * atom_num, 2*self.emb_dim)
        #print(h.shape)
        # Edge features
        vectors = batch.pos[batch.edge_index[0]] - batch.pos[batch.edge_index[1]]  # [n_edges, 3]
        lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)  # [n_edges, 1]

        edge_sh = self.spherical_harmonics(vectors)
        edge_feats = self.radial_embedding(lengths)
        
        for conv, reshape, prod in zip(self.convs, self.reshapes, self.prods):
            # Message passing layer
            h_update = conv(h, batch.edge_index, edge_sh, edge_feats)
            
            # Update node features
            sc = F.pad(h, (0, h_update.shape[-1] - h.shape[-1]))
            h = prod(reshape(h_update), sc, None)

        out = self.pool(h, batch.batch)  # (n, d) -> (batch_size, d)
        
        if not self.equivariant_pred:
            # Select only scalars for invariant prediction
            out = out[:,:self.emb_dim]
        
        return self.pred(out)  # (batch_size, out_dim)
