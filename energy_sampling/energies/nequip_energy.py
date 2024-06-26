
from ml_collections import ConfigDict
from energies.model.nequip import NequIPEnergyModel, model_from_config
from models.utils import smiles2graph
import torch
from torch.utils import dlpack as torch_dlpack
import pickle
from .base_set import BaseSet
import jax
from jax import dlpack as jax_dlpack
import jax.numpy as jnp
from jax.tree_util import tree_map

from inspect import signature
from functools import wraps
from flax import serialization


def j2t(x_jax):
    x_torch = torch_dlpack.from_dlpack(jax_dlpack.to_dlpack(x_jax))
    return x_torch

def t2j(x_torch):
    x_torch = x_torch.contiguous() # https://github.com/google/jax/issues/8082
    x_jax = jax_dlpack.from_dlpack(torch_dlpack.to_dlpack(x_torch))
    return x_jax

def tree_t2j(x_torch):
    return tree_map(lambda t: t2j(t) if isinstance(t, torch.Tensor) else t, x_torch)

def tree_j2t(x_jax):
    return tree_map(lambda t: j2t(t) if isinstance(t, jnp.ndarray) else t, x_jax)

def jax2torch(fn):
    @wraps(fn)
    def inner(*args, **kwargs):
        class JaxFun(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *args):
                args = tree_t2j(args)
                y_, ctx.fun_vjp = jax.vjp(fn, *args)
                return tree_j2t(y_)

            @staticmethod
            def backward(ctx, *grad_args):
                grad_args = tree_t2j(grad_args) if len(grad_args) > 1 else t2j(grad_args[0])
                grads = ctx.fun_vjp(grad_args)
                grads = tuple(map(lambda t: t if isinstance(t, jnp.ndarray) else None, grads))
                return tree_j2t(grads)

        sig = signature(fn)
        bound = sig.bind(*args, **kwargs)
        bound.apply_defaults()
        return JaxFun.apply(*bound.arguments.values())
    return inner

def initialize_nequip_cfg_MaxSetup(n_species, r_cut) -> ConfigDict:
    """Initialize configuration based on values of original paper."""

    config = ConfigDict()

    # Information on hyperparameters
    # 1. Cutoff is very important - For MD17 papers employ cutoff = 4 Å
    # Further potential changes:
        # 1. Increase hidden layers
        # 2. Increase number of basis functions

    # network
    config.graph_net_steps = 5  #2  #(3-5 work best -> Paper)
    config.nonlinearities = {'e': 'raw_swish', 'o': 'tanh'}
    config.use_sc = True
    config.n_elements = n_species
    config.hidden_irreps = '64x0o + 64x0e + 64x1o + 64x1e + 64x2o + 64x2e'  # l_max=2, parity=True
    config.sh_irreps = '1x0e+1x1o+1x2e'     # l_max=2, parity=True
    config.num_basis = 8
    config.r_max = r_cut
    config.radial_net_nonlinearity = 'raw_swish'
    config.radial_net_n_hidden = 64  # 8  # Smaller is faster # 64
    config.radial_net_n_layers = 3

    # Setting dependend on dataset
    # QM7x all data
    config.n_neighbors = 15
    config.shift = 0.
    config.scale = 1.
    config.scalar_mlp_std = 4.
    return config

class NequipEnergy(BaseSet):

    def __init__(self, local_model, smiles):
        graph = smiles2graph(smiles_string=smiles)
        config = initialize_nequip_cfg_MaxSetup(n_species=graph['num_nodes'], r_cut=10)
        energy_jax = model_from_config(config)

        with open(local_model, 'rb') as f:
            weights = pickle.load(f)

        energy_jax = serialization.from_state_dict(energy_jax, weights)

        self.energy_model = jax2torch(energy_jax)
        self.data_ndim = 3*graph['num_nodes']
        print(self.data_ndim)

    def energy(self, xyz):
        return self.energy_model(xyz)

    def sample(self, batch_size):
        return None