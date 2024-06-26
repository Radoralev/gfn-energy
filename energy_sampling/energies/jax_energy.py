import torch
import torch.nn as nn
import pickle
import jax
import jax.numpy as jnp
from flax import traverse_util

def initialize_nequip_cfg_MaxSetup(n_species, r_cut):
    """Initialize configuration based on values of original paper."""

    config = {}

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

class PyTorchJAXModel(nn.Module):
    def __init__(self, jax_model_config, weights_path):
        super(PyTorchJAXModel, self).__init__()
        
        # Load JAX model config
        self.jax_model_config = jax_model_config
        
        # Convert JAX model config to PyTorch
        self.pytorch_params = self.convert_jax_to_pytorch(jax_model_config)
        
        # Initialize PyTorch model layers
        self.init_pytorch_layers(self.pytorch_params)
        
        # Load pre-trained weights
        self.load_weights(weights_path)
        
    def convert_jax_to_pytorch(self, jax_params):
        # Flatten the JAX model parameters
        flat_jax_params = traverse_util.flatten_dict(jax_params)
        
        # Convert to PyTorch tensors
        pytorch_params = {}
        for key, value in flat_jax_params.items():
            pytorch_params[key] = torch.from_numpy(value)
            
        return pytorch_params
    
    def init_pytorch_layers(self, pytorch_params):
        # Initialize PyTorch layers based on the flattened parameters
        # Example: Assuming a simple linear model for demonstration
        self.layers = nn.ModuleList()
        for key, param in pytorch_params.items():
            if 'weight' in key:
                in_features, out_features = param.shape
                layer = nn.Linear(in_features, out_features)
                self.layers.append(layer)
    
    def load_weights(self, weights_path):
        # Load pre-trained weights from a pickle file
        with open(weights_path, 'rb') as f:
            jax_weights = pickle.load(f)
        
        # Convert JAX weights to PyTorch tensors
        pytorch_weights = self.convert_jax_to_pytorch(jax_weights)
        
        # Load weights into the PyTorch model
        for layer, (key, param) in zip(self.layers, pytorch_weights.items()):
            if 'weight' in key:
                layer.weight.data = param
            elif 'bias' in key:
                layer.bias.data = param
    
    def forward(self, x):
        # Define the forward pass
        for layer in self.layers:
            x = layer(x)
        return x

# Example usage
jax_model_config = initialize_nequip_cfg_MaxSetup(n_species=5, r_cut=4.0)
weights_path = 'path/to/jax_weights.pkl'

model = PyTorchJAXModel(jax_model_config, weights_path)
input_data = torch.tensor([[1.0, 2.0]])
output = model(input_data)
print(output)

