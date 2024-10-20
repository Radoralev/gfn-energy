import torch
import numpy as np
from einops import rearrange
from torch import nn
import math

from .mace import MACEModel
from .attention import AtomAttention
from .utils import smiles2graph, prep_input
from torch_geometric.data import Batch
from torch_geometric import loader 
from models.layers.egnn_layer import EGNNLayer

class TimeConder(nn.Module):
    def __init__(self, channel, out_dim, num_layers):
        super().__init__()
        self.register_buffer(
            "timestep_coeff", torch.linspace(start=0.1, end=100, steps=channel)[None]
        )
        self.timestep_phase = nn.Parameter(torch.randn(channel)[None])
        self.layers = nn.Sequential(
            nn.Linear(2 * channel, channel),
            *[
                nn.Sequential(
                    nn.GELU(),
                    nn.Linear(channel, channel),
                )
                for _ in range(num_layers - 1)
            ],
            nn.GELU(),
            nn.Linear(channel, out_dim)
        )

        self.layers[-1].weight.data.fill_(0.0)
        self.layers[-1].bias.data.fill_(0.01)

    def forward(self, t):
        sin_cond = torch.sin((self.timestep_coeff * t.float()) + self.timestep_phase)
        cos_cond = torch.cos((self.timestep_coeff * t.float()) + self.timestep_phase)
        cond = rearrange([sin_cond, cos_cond], "d b w -> b (d w)")
        return self.layers(cond)


class FourierMLP(nn.Module):
    def __init__(
            self,
            in_shape=2,
            out_shape=2,
            num_layers=2,
            channels=128,
            zero_init=True,
    ):
        super().__init__()

        self.in_shape = (in_shape,)
        self.out_shape = (out_shape,)

        self.register_buffer(
            "timestep_coeff", torch.linspace(start=0.1, end=100, steps=channels)[None]
        )
        self.timestep_phase = nn.Parameter(torch.randn(channels)[None])
        self.input_embed = nn.Linear(int(np.prod(in_shape)), channels)
        self.timestep_embed = nn.Sequential(
            nn.Linear(2 * channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.layers = nn.Sequential(
            nn.GELU(),
            *[
                nn.Sequential(nn.Linear(channels, channels), nn.GELU())
                for _ in range(num_layers)
            ],
            nn.Linear(channels, int(np.prod(self.out_shape))),
        )
        if zero_init:
            self.layers[-1].weight.data.fill_(0.0)
            self.layers[-1].bias.data.fill_(0.0)

    def forward(self, cond, inputs):
        cond = cond.view(-1, 1).expand((inputs.shape[0], 1))
        sin_embed_cond = torch.sin(
            (self.timestep_coeff * cond.float()) + self.timestep_phase
        )
        cos_embed_cond = torch.cos(
            (self.timestep_coeff * cond.float()) + self.timestep_phase
        )
        embed_cond = self.timestep_embed(
            rearrange([sin_embed_cond, cos_embed_cond], "d b w -> b (d w)")
        )
        embed_ins = self.input_embed(inputs.view(inputs.shape[0], -1))
        out = self.layers(embed_ins + embed_cond)
        return out.view(-1, *self.out_shape)


class TimeEncoding(nn.Module):
    def __init__(self, harmonics_dim: int, dim: int, hidden_dim: int = 64):
        super(TimeEncoding, self).__init__()

        pe = torch.arange(1, harmonics_dim + 1).float().unsqueeze(0) * 2 * math.pi
        self.t_model = nn.Sequential(
            nn.Linear(2 * harmonics_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
            nn.GELU()
        )
        self.register_buffer('pe', pe)

    def forward(self, t: float = None):
        """
        Arguments:
            t: float
        """
        t_sin = (t * self.pe).sin()
        t_cos = (t * self.pe).cos()
        t_emb = torch.cat([t_sin, t_cos], dim=-1)
        return self.t_model(t_emb)


class StateEncoding(nn.Module):
    def __init__(self, s_dim: int, hidden_dim: int = 64, s_emb_dim: int = 64):
        super(StateEncoding, self).__init__()

        self.x_model = nn.Sequential(
            nn.Linear(s_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, s_emb_dim),
            nn.GELU()
        )

    def forward(self, s):
        return self.x_model(s)

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
        num_atom_features: int = 1
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
        # Embedding lookup for initial node features
        self.num_atom_features = num_atom_features
        self.embedding = torch.nn.Embedding(num_atom_features+1, emb_dim) 
        self.in_dim = in_dim
        self.emb_dim = emb_dim
        self.out_dim = out_dim
        # Stack of GNN layers
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(EGNNLayer(emb_dim, activation, norm, aggr))


    
    def forward(self, batch, t=None):
        h = self.embedding(batch.atoms[..., 0])
        if t is not None:
            # match h shape
            t = t.repeat(h.shape[0]//t.shape[0], 1)
            h = h + t
        #TODO batch size is hardcoded here 
        #h = h.view(-1, batch.atoms.shape[0]//32, self.emb_dim)
        # print(batch.pos.shape/, h.shape)
        pos = batch.pos.to(dtype=torch.float32)#.reshape(-1, 3)
        for conv in self.convs:
            # Message passing layer
            h_update, pos_update = conv(h, pos, batch.edge_index)

            # Update node features (n, d) -> (n, d)
            h = h + h_update if self.residual else h_update 

            # Update node coordinates (no residual) (n, 3) -> (n, 3)
            pos = pos_update
        return pos.view(-1, self.out_dim)

class EquivariantPolicy(nn.Module):
    def __init__(self, model: str = 'egnn', in_dim: int = 32, t_dim: int = 32, hidden_dim: int = 64, out_dim: int = None, num_layers: int = 2, smiles: str = None, zero_init: bool = False, model_args: dict = None):
        super(EquivariantPolicy, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.graph = smiles2graph(smiles)
        self.in_dim = in_dim
        if model == 'mace':
            self.model = MACEModel(in_dim=in_dim, out_dim=out_dim, mlp_dim=hidden_dim, emb_dim=t_dim, equivariant_pred=True, num_layers=num_layers,).to(self.device)
            if zero_init:
                self.model.pred.weight.data.fill_(0.0)
                self.model.pred.bias.data.fill_(0.0)
        elif model == 'egnn':
            self.model = EGNNModel(in_dim=in_dim, emb_dim=hidden_dim, out_dim=out_dim, num_layers=num_layers, num_atom_features=max(self.graph['node_feat'][:, 0]), equivariant_pred=True)
        # data_list = prep_input(self.graph, pos=torch.ones(64, self.in_dim//3, 3) ,device=self.device)
        # self.batch_train = Batch.from_data_list(data_list)
        # data_list = prep_input(self.graph, pos=torch.ones(512, self.in_dim//3, 3) ,device=self.device)
        # self.batch_val = Batch.from_data_list(data_list)
        # data_list = prep_input(self.graph, pos=torch.ones(2048, self.in_dim//3, 3) ,device=self.device)
        # self.batch_final_val = Batch.from_data_list(data_list)

    def forward(self, s, t):
        data_list = prep_input(self.graph, pos=s.reshape(-1, self.in_dim//3, 3), device=self.device)
        batch = Batch.from_data_list(data_list)
      #  batch.pos = s.reshape(-1, 3)
        return self.model(batch, t)


class AttentionPolicy(nn.Module):
    def __init__(self, in_dim: int = 32, t_dim: int = 32, hidden_dim: int = 64, out_dim: int = None, num_layers: int = 2, smiles: str = None, zero_init: bool = False):
        super(AttentionPolicy, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.graph = smiles2graph(smiles)
        self.in_dim = in_dim
        self.model = AtomAttention(embeddings_dim=3, output_dim=out_dim, t_dim=t_dim, kernel_size=self.graph['num_nodes']).to(self.device)
        if zero_init:
            self.model.output.net[-1].weight.data.fill_(0.0)
            self.model.output.net[-1].bias.data.fill_(0.0)
        self.atom_types = torch.from_numpy(self.graph['node_feat'][:, 0]).to(self.device)

    def forward(self, s, t):
        return self.model(s, t, self.atom_types)


class JointPolicy(nn.Module):
    def __init__(self, s_dim: int, s_emb_dim: int, t_dim: int, hidden_dim: int = 64, out_dim: int = None,
                 zero_init: bool = False, model = None, smiles=None):
        super(JointPolicy, self).__init__()
        if out_dim is None:
            out_dim = 2 * s_dim


        if model == 'mlp':
            self.model = nn.Sequential(
                nn.Linear(s_emb_dim + t_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, out_dim)
            )
            if zero_init:
                self.model[-1].weight.data.fill_(0.0)
                self.model[-1].bias.data.fill_(0.0)

    def forward(self, s, t):
        return self.model(torch.cat([s, t], dim=-1))


class FlowModel(nn.Module):
    def __init__(self, s_emb_dim: int, t_dim: int, hidden_dim: int = 64, out_dim: int = 1):
        super(FlowModel, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(s_emb_dim + t_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, s, t):
        return self.model(torch.cat([s, t], dim=-1))


class LangevinScalingModel(nn.Module):
    def __init__(self, s_emb_dim: int, t_dim: int, hidden_dim: int = 64, out_dim: int = 1, zero_init: bool = False):
        super(LangevinScalingModel, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(s_emb_dim + t_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

        if zero_init:
            self.model[-1].weight.data.fill_(0.0)
            self.model[-1].bias.data.fill_(0.01)

    def forward(self, s, t):
        return self.model(torch.cat([s, t], dim=-1))


class TimeEncodingPIS(nn.Module):
    def __init__(self, harmonics_dim: int, dim: int, hidden_dim: int = 64):
        super(TimeEncodingPIS, self).__init__()

        pe = torch.linspace(start=0.1, end=100, steps=harmonics_dim)[None]

        self.timestep_phase = nn.Parameter(torch.randn(harmonics_dim)[None])

        self.t_model = nn.Sequential(
            nn.Linear(2 * harmonics_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.register_buffer('pe', pe)

    def forward(self, t: float = None):
        """
        Arguments:
            t: float
        """
        t_sin = ((t * self.pe) + self.timestep_phase).sin()
        t_cos = ((t * self.pe) + self.timestep_phase).cos()
        t_emb = torch.cat([t_sin, t_cos], dim=-1)
        return self.t_model(t_emb)


class StateEncodingPIS(nn.Module):
    def __init__(self, s_dim: int, hidden_dim: int = 64, s_emb_dim: int = 64):
        super(StateEncodingPIS, self).__init__()

        self.x_model = nn.Linear(s_dim, s_emb_dim)

    def forward(self, s):
        return self.x_model(s)


class JointPolicyPIS(nn.Module):
    def __init__(self, s_dim: int, s_emb_dim: int, t_dim: int, hidden_dim: int = 64, out_dim: int = None,
                 num_layers: int = 2,
                 zero_init: bool = False):
        super(JointPolicyPIS, self).__init__()
        if out_dim is None:
            out_dim = 2 * s_dim

        assert s_emb_dim == t_dim, print("Dimensionality of state embedding and time embedding should be the same!")

        self.model = nn.Sequential(
            nn.GELU(),
            *[
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())
                for _ in range(num_layers)
            ],
            nn.Linear(hidden_dim, out_dim),
        )

        if zero_init:
            self.model[-1].weight.data.fill_(0.0)
            self.model[-1].bias.data.fill_(0.0)

    def forward(self, s, t):
        return self.model(s + t)


class FlowModelPIS(nn.Module):
    def __init__(self, s_dim: int, s_emb_dim: int, t_dim: int, hidden_dim: int = 64, out_dim: int = 1,
                 num_layers: int = 2,
                 zero_init: bool = False):
        super(FlowModelPIS, self).__init__()

        assert s_emb_dim == t_dim, print("Dimensionality of state embedding and time embedding should be the same!")

        self.model = nn.Sequential(
            nn.GELU(),
            *[
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU())
                for _ in range(num_layers)
            ],
            nn.Linear(hidden_dim, out_dim),
        )

        if zero_init:
            self.model[-1].weight.data.fill_(0.0)
            self.model[-1].bias.data.fill_(0.0)

    def forward(self, s, t):
        return self.model(s + t)


class LangevinScalingModelPIS(nn.Module):
    def __init__(self, s_emb_dim: int, t_dim: int, hidden_dim: int = 64, out_dim: int = 1, num_layers: int = 3,
                 zero_init: bool = False):
        super(LangevinScalingModelPIS, self).__init__()

        pe = torch.linspace(start=0.1, end=100, steps=t_dim)[None]

        self.timestep_phase = nn.Parameter(torch.randn(t_dim)[None])

        self.lgv_model = nn.Sequential(
            nn.Linear(2 * t_dim, hidden_dim),
            *[
                nn.Sequential(
                    nn.GELU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
                for _ in range(num_layers - 1)
            ],
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim)
        )

        self.register_buffer('pe', pe)

        if zero_init:
            self.lgv_model[-1].weight.data.fill_(0.0)
            self.lgv_model[-1].bias.data.fill_(0.01)

    def forward(self, t):
        t_sin = ((t * self.pe) + self.timestep_phase).sin()
        t_cos = ((t * self.pe) + self.timestep_phase).cos()
        t_emb = torch.cat([t_sin, t_cos], dim=-1)
        return self.lgv_model(t_emb)