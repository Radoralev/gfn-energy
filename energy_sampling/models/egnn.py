import torch
from torch.nn import functional as F
from torch_geometric.nn import global_add_pool, global_mean_pool

from models.layers.egnn_layer import EGNNLayer
from models.mace import prep_input, smiles2graph

class EGNNModel(torch.nn.Module):
    """
    E-GNN model from "E(n) Equivariant Graph Neural Networks".
    """
    def __init__(
        self,
        num_layers: int = 5,
        emb_dim: int = 128,
        in_dim: int = 1,
        out_dim: int = 1,
        activation: str = "relu",
        norm: str = "layer",
        aggr: str = "sum",
        pool: str = "sum",
        residual: bool = True,
        equivariant_pred: bool = False,
        smiles: str = None
    ):
        """
        Initializes an instance of the EGNNModel class with the provided parameters.

        Parameters:
        - num_layers (int): Number of layers in the model (default: 5)
        - emb_dim (int): Dimension of the node embeddings (default: 128)
        - in_dim (int): Input dimension of the model (default: 1)
        - out_dim (int): Output dimension of the model (default: 1)
        - activation (str): Activation function to be used (default: "relu")
        - norm (str): Normalization method to be used (default: "layer")
        - aggr (str): Aggregation method to be used (default: "sum")
        - pool (str): Global pooling method to be used (default: "sum")
        - residual (bool): Whether to use residual connections (default: True)
        - equivariant_pred (bool): Whether it is an equivariant prediction task (default: False)
        """
        super().__init__()
        self.equivariant_pred = equivariant_pred
        self.residual = residual
        self.smiles = smiles
        self.smiles_graph = smiles2graph(smiles)
        # Embedding lookup for initial node features
        self.emb_in = torch.nn.Embedding(in_dim, emb_dim)
        
        self.down = torch.nn.Linear(2*emb_dim, emb_dim)
        
        self.in_dim = in_dim
        self.emb_dim = emb_dim
        # Stack of GNN layers
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(EGNNLayer(emb_dim, activation, norm, aggr))

        # Global pooling/readout function
        self.pool = {"mean": global_mean_pool, "sum": global_add_pool}[pool]

        if self.equivariant_pred:
            # Linear predictor for equivariant tasks using geometric features
            self.pred = torch.nn.Linear(emb_dim + 3, out_dim)
        else:
            # MLP predictor for invariant tasks using only scalar features
            self.pred = torch.nn.Sequential(
                torch.nn.Linear(emb_dim, emb_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(emb_dim, out_dim)
            )

    def forward(self, x):
        
        pos, t = x[:, :self.in_dim], x[:, self.in_dim:]
        pos = pos.reshape(-1, int(self.in_dim / 3), 3)
        bs, atom_num, _ = pos.shape
        batch = prep_input(self.smiles_graph, pos, device=self.emb_in.weight.device)
        # Node embedding
        #atom_number = batch.num_nodes
        h = self.emb_in(batch.atoms)  # (n,) -> (n, d)
        h = h.view(bs, atom_num, self.emb_dim)
        t = t.unsqueeze(1).expand(h.size(0), h.size(1), h.size(2))
        h = torch.cat([h, t], dim=-1)

        h = self.down(h)
        h = h.view(bs * atom_num, self.emb_dim)
            
        pos = batch.pos  # (n, 3)

        for conv in self.convs:
            # Message passing layer
            h_update, pos_update = conv(h, pos, batch.edge_index)

            # Update node features (n, d) -> (n, d)
            h = h + h_update if self.residual else h_update 

            # Update node coordinates (no residual) (n, 3) -> (n, 3)
            pos = pos_update
    
        if not self.equivariant_pred:
            # Select only scalars for invariant prediction
            out = self.pool(h, batch.batch)  # (n, d) -> (batch_size, d)
        else:
            out = self.pool(torch.cat([h, pos], dim=-1), batch.batch)
            
        return self.pred(out)  # (batch_size, out_dim)
