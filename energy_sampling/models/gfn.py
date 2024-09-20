import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from .architectures import *
from utils import gaussian_params
from .mace import MACEModel
logtwopi = math.log(2 * math.pi)

from torch.distributions import Normal, VonMises, MixtureSameFamily 


def compute_wrapped_gaussian_log_prob(delta_s, mean, std):
    # Approximate the wrapped Gaussian log probability using a finite sum
    # delta_s: Difference in states, adjusted for wrapping (shape: [batch_size, dim])
    # mean: Mean of the Gaussian (shape: [batch_size, dim])
    # std: Standard deviation of the Gaussian (shape: [batch_size, dim])

    # Parameters
    N = 10  # Number of terms to include in the sum
    two_pi = 2 * np.pi

    # Initialize log probability tensor
    log_probs = None  # Will be initialized in the loop

    # Prepare the range of k values for summation
    ks = torch.arange(-N, N + 1, device=delta_s.device).reshape(1, 1, 2 * N + 1)

    # Expand tensors to match dimensions for broadcasting
    delta_s_expanded = delta_s.unsqueeze(2) + ks * two_pi
    mean_expanded = mean.unsqueeze(2)
    std_expanded = std.unsqueeze(2)

    # Compute the normal log probabilities for each wrapped component
    exponent = -0.5 * ((delta_s_expanded - mean_expanded) / std_expanded) ** 2
    log_coeff = -torch.log(std_expanded * np.sqrt(2 * np.pi))

    # Sum over k in log space using logsumexp
    log_prob_components = exponent + log_coeff
    log_probs = torch.logsumexp(log_prob_components, dim=2)

    # Sum over dimensions if necessary
    log_prob = log_probs.sum(dim=1)  # Sum over state dimensions

    return log_prob

class GFN(nn.Module):
    def __init__(self, dim: int, s_emb_dim: int, hidden_dim: int,
                 harmonics_dim: int, t_dim: int, log_var_range: float = 4.,
                 t_scale: float = 1., langevin: bool = False, learned_variance: bool = True,
                 trajectory_length: int = 100, partial_energy: bool = False,
                 clipping: bool = False, lgv_clip: float = 1e2, gfn_clip: float = 1e4, pb_scale_range: float = 1.,
                 langevin_scaling_per_dimension: bool = True, conditional_flow_model: bool = False,
                 learn_pb: bool = False, model='mlp', smiles: str = '',
                 pis_architectures: bool = False, lgv_layers: int = 3, joint_layers: int = 2, model_args: dict = {},
                 zero_init: bool = False, device=torch.device('cuda'), equivariant_architectures: bool = False, energy = None):
        super(GFN, self).__init__()
        self.dim = dim
        self.harmonics_dim = harmonics_dim
        self.t_dim = t_dim
        self.s_emb_dim = s_emb_dim
        self.smiles = smiles
        self.model = model
        self.model_args = model_args
        self.trajectory_length = trajectory_length
        self.langevin = langevin
        self.learned_variance = learned_variance
        self.partial_energy = partial_energy
        self.t_scale = t_scale
        self.energy = energy

        self.clipping = clipping
        self.lgv_clip = lgv_clip
        self.gfn_clip = gfn_clip

        self.langevin_scaling_per_dimension = langevin_scaling_per_dimension
        self.conditional_flow_model = conditional_flow_model
        self.learn_pb = learn_pb
        self.equivariant_architectures = equivariant_architectures
        self.pis_architectures = pis_architectures
        self.lgv_layers = lgv_layers
        self.joint_layers = joint_layers

        self.pf_std_per_traj = np.sqrt(self.t_scale)
        self.dt = 1. / trajectory_length
        self.log_var_range = log_var_range
        self.device = device

        if self.equivariant_architectures:
            self.t_model = TimeEncodingPIS(harmonics_dim, t_dim, hidden_dim)
            self.s_model = StateEncodingPIS(dim, hidden_dim, s_emb_dim)
            self.joint_model = EquivariantPolicy(
                model=model, 
                in_dim=dim, 
                t_dim=t_dim, 
                hidden_dim=hidden_dim, 
                out_dim=dim, 
                num_layers=joint_layers, 
                smiles=smiles, 
                zero_init=zero_init,
                model_args=model_args)
            if learn_pb:
                self.back_model = EquivariantPolicy(
                model=model, 
                in_dim=dim, 
                t_dim=t_dim, 
                hidden_dim=hidden_dim, 
                out_dim=2 * dim, 
                num_layers=joint_layers, 
                smiles=smiles, 
                zero_init=zero_init,
                model_args=model_args)
            self.pb_scale_range = pb_scale_range
            if self.conditional_flow_model:
                self.flow_model = FlowModel(dim, t_dim, hidden_dim, 1)
            else:
                self.flow_model = torch.nn.Parameter(torch.tensor(0.).to(self.device))

            if self.langevin_scaling_per_dimension:
                self.langevin_scaling_model = LangevinScalingModelPIS(s_emb_dim, t_dim, hidden_dim, dim,
                                                                      lgv_layers, zero_init)
            else:
                self.langevin_scaling_model = LangevinScalingModelPIS(s_emb_dim, t_dim, hidden_dim, 1,
                                                                      lgv_layers, zero_init)
        elif self.pis_architectures:

            self.t_model = TimeEncodingPIS(harmonics_dim, t_dim, hidden_dim)
            self.s_model = StateEncodingPIS(dim, hidden_dim, s_emb_dim)
            if model == 'mlp':
                self.joint_model = JointPolicyPIS(dim, s_emb_dim, t_dim, hidden_dim, 2*dim, joint_layers, zero_init)
                if learn_pb:
                    self.back_model = JointPolicyPIS(dim, s_emb_dim, t_dim, hidden_dim, 2*dim, joint_layers, zero_init)
            elif model == 'attention':
                self.joint_model = AttentionPolicy(in_dim=dim, t_dim=t_dim, hidden_dim=hidden_dim, out_dim=2*dim, num_layers=joint_layers, zero_init=zero_init, smiles=smiles)
                if learn_pb:
                    self.back_model = AttentionPolicy(in_dim=dim, t_dim=t_dim, hidden_dim=hidden_dim, out_dim=2*dim, num_layers=joint_layers, zero_init=zero_init, smiles=smiles)
            self.pb_scale_range = pb_scale_range

            if self.conditional_flow_model:
                if model == 'attention':
                    self.flow_model = AttentionPolicy(in_dim=dim, t_dim=t_dim, hidden_dim=hidden_dim, out_dim=1, num_layers=joint_layers, zero_init=False, smiles=smiles)#
                else:
                    self.flow_model = FlowModelPIS(dim, s_emb_dim, t_dim, hidden_dim, 1, joint_layers)
            else:
                self.flow_model = torch.nn.Parameter(torch.tensor(0.).to(self.device))

            if self.langevin_scaling_per_dimension:
                self.langevin_scaling_model = LangevinScalingModelPIS(s_emb_dim, t_dim, hidden_dim, dim,
                                                                      lgv_layers, zero_init)
            else:
                self.langevin_scaling_model = LangevinScalingModelPIS(s_emb_dim, t_dim, hidden_dim, 1,
                                                                      lgv_layers, zero_init)

        else:

            self.t_model = TimeEncoding(harmonics_dim, t_dim, hidden_dim)
            self.s_model = StateEncoding(dim, hidden_dim, s_emb_dim)
            self.joint_model = JointPolicy(dim, s_emb_dim, t_dim, hidden_dim, 2 * dim, zero_init, model=model, smiles=smiles).to(device)
            if learn_pb:
                self.back_model = JointPolicy(dim, s_emb_dim, t_dim, hidden_dim, 2 * dim, zero_init).to(device)
            self.pb_scale_range = pb_scale_range

            if self.conditional_flow_model:
                self.flow_model = FlowModel(s_emb_dim, t_dim, hidden_dim, 1)
            else:
                self.flow_model = torch.nn.Parameter(torch.tensor(0.).to(self.device))

            if self.langevin_scaling_per_dimension:
                self.langevin_scaling_model = LangevinScalingModel(s_emb_dim, t_dim, hidden_dim, dim, zero_init)
            else:
                self.langevin_scaling_model = LangevinScalingModel(s_emb_dim, t_dim, hidden_dim, 1, zero_init)
        # print number parameters
        print('Number of parameters: ', self.num_parameters())

    def split_params(self, tensor):
        if self.equivariant_architectures and self.model =='egnn':
            return tensor[..., :self.dim], torch.ones_like(tensor[..., :self.dim]) + np.log(self.pf_std_per_traj) * 2.
        mean, logvar = gaussian_params(tensor)
        if not self.learned_variance:
            logvar = torch.zeros_like(logvar)
        else:
            logvar = torch.tanh(logvar) * self.log_var_range
        return mean, logvar + np.log(self.pf_std_per_traj) * 2.

    def predict_next_state(self, s, t, log_r):
        if self.langevin:
            s.requires_grad_(True)
            with torch.enable_grad():
                grad_log_r = torch.autograd.grad(log_r(s).sum(), s)[0].detach()
                grad_log_r = torch.nan_to_num(grad_log_r)
                if self.clipping:
                    grad_log_r = torch.clip(grad_log_r, -self.lgv_clip, self.lgv_clip)

        bsz = s.shape[0]

        t_lgv = t

        t = self.t_model(t).repeat(bsz, 1)
        if self.model == 'mlp':
            s = self.s_model(s)
        s_new = self.joint_model(s, t)

        flow = self.flow_model(s, t).squeeze(-1) if self.conditional_flow_model or self.partial_energy else self.flow_model # type: ignore

        if self.langevin:
            if self.pis_architectures or self.equivariant_architectures:
                scale = self.langevin_scaling_model(t_lgv)
            else:
                scale = self.langevin_scaling_model(s, t)
            s_new[..., :self.dim] += scale * grad_log_r

        if self.clipping:
            s_new = torch.clip(s_new, -self.gfn_clip, self.gfn_clip)
        return s_new, flow.squeeze(-1)

    
    
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_trajectory_fwd(self, s, exploration_std, log_r):
        bsz = s.shape[0]
        logpf = torch.zeros((bsz, self.trajectory_length), device=self.device)
        logpb = torch.zeros((bsz, self.trajectory_length), device=self.device)
        logf = torch.zeros((bsz, self.trajectory_length + 1), device=self.device)
        states = torch.zeros((bsz, self.trajectory_length + 1, self.dim), device=self.device)
        states[:, 0] = s

        # Define constants for wrapping
        two_pi = 2 * np.pi
        pi = np.pi

        for i in range(self.trajectory_length):
            # Predict next state using neural nets (policies and flow model)
            pfs, flow = self.predict_next_state(s, i * self.dt, log_r)
            pf_mean, pflogvars = self.split_params(pfs)

            logf[:, i] = flow

            if exploration_std is None:
                pflogvars_sample = pflogvars.detach()
            else:
                expl = exploration_std(i)
                if expl <= 0.0:
                    pflogvars_sample = pflogvars.detach()
                else:
                    add_log_var = torch.full_like(pflogvars, np.log(exploration_std(i) / np.sqrt(self.dt)) * 2)
                    pflogvars_sample = torch.logaddexp(pflogvars, add_log_var).detach()

            # Sampling from the wrapped Gaussian distribution
            # Sample from standard normal
            epsilon = torch.randn_like(s, device=self.device)

            # Compute variance term
            std = (pflogvars_sample / 2).exp()
            var_term = np.sqrt(self.dt) * std * epsilon

            # Compute the next state before wrapping
            s_next = s + self.dt * pf_mean.detach() + var_term

            # Wrap the next state to the interval [-π, π)
            s_wrapped = ((s_next + pi) % two_pi) - pi

            # Calculate the noise for log probability computation
            # Adjusted for the wrapped Gaussian
            delta_s = ((s_wrapped - s + pi) % two_pi) - pi
            mean_delta = self.dt * pf_mean
            std_delta = np.sqrt(self.dt) * std

            # Compute the wrapped Gaussian log probability
            logpf[:, i] = compute_wrapped_gaussian_log_prob(delta_s, mean_delta, std_delta)

            # Update state
            s = s_wrapped
            states[:, i + 1] = s

            # Backward probability computation (if applicable)
            if self.learn_pb and i > 0:
                # Similar adjustments for the backward process
                # Predict backward mean and variance corrections
                t = self.t_model((i + 1) * self.dt).repeat(bsz, 1)
                pbs = self.back_model(self.s_model(s), t)
                dmean, dvar = self.split_params(pbs)
                back_mean_correction = 1 + dmean.tanh() * self.pb_scale_range
                back_var_correction = 1 + dvar.tanh() * self.pb_scale_range
            else:
                back_mean_correction = torch.ones_like(s)
                back_var_correction = torch.ones_like(s)

            if i > 0:
                # Compute backward mean and variance
                back_mean = s - self.dt * s / ((i + 1) * self.dt) * back_mean_correction
                back_var = (self.pf_std_per_traj ** 2) * self.dt * i / (i + 1) * back_var_correction

                # Adjust for wrapping
                delta_s_backward = ((states[:, i] - s + pi) % two_pi) - pi
                noise_backward = delta_s_backward / back_var.sqrt()

                # Compute wrapped Gaussian log probability for backward transition
                logpb[:, i] = compute_wrapped_gaussian_log_prob(delta_s_backward, back_mean - s, back_var.sqrt())

        return states, logpf, logpb, logf


    def get_trajectory_bwd(self, s, exploration_std, log_r):
        bsz = s.shape[0]

        logpf = torch.zeros((bsz, self.trajectory_length), device=self.device)
        logpb = torch.zeros((bsz, self.trajectory_length), device=self.device)
        logf = torch.zeros((bsz, self.trajectory_length + 1), device=self.device)
        states = torch.zeros((bsz, self.trajectory_length + 1, self.dim), device=self.device)
        states[:, -1] = s

        for i in range(self.trajectory_length):
            if i < self.trajectory_length - 1:
                if self.learn_pb:
                    t = self.t_model(1. - i * self.dt).repeat(bsz, 1)
                    pbs = self.back_model(self.s_model(s), t)
                    dmean, dvar = gaussian_params(pbs)
                    back_mean_correction = 1 + dmean.tanh() * self.pb_scale_range
                    back_var_correction = 1 + dvar.tanh() * self.pb_scale_range
                else:
                    back_mean_correction, back_var_correction = torch.ones_like(s), torch.ones_like(s)

                mean = s - self.dt * s / (1. - i * self.dt) * back_mean_correction
                var = ((self.pf_std_per_traj ** 2) * self.dt * (1. - (i + 1) * self.dt)) / (
                            1 - i * self.dt) * back_var_correction
                s_ = mean.detach() + var.sqrt().detach() * torch.randn_like(s, device=self.device)
                noise_backward = (s_ - mean) / var.sqrt()
                logpb[:, self.trajectory_length - i - 1] = -0.5 * (noise_backward ** 2 + logtwopi + var.log()).sum(1)
            else:
                s_ = torch.zeros_like(s)

            pfs, flow = self.predict_next_state(s_, (1. - (i + 1) * self.dt), log_r)
            pf_mean, pflogvars = self.split_params(pfs)

            logf[:, self.trajectory_length - i - 1] = flow
            if self.partial_energy:
                ref_log_var = np.log(self.t_scale * max(1, self.trajectory_length - i - 1) * self.dt)
                log_p_ref = -0.5 * (logtwopi + ref_log_var + np.exp(-ref_log_var) * (s ** 2)).sum(1)
                logf[:, self.trajectory_length - i - 1] += (i + 1) * self.dt * log_p_ref + (
                        self.trajectory_length - i - 1) * self.dt * log_r(s)

            noise = ((s - s_) - self.dt * pf_mean) / (np.sqrt(self.dt) * (pflogvars / 2).exp())
            logpf[:, self.trajectory_length - i - 1] = -0.5 * (noise ** 2 + logtwopi + np.log(self.dt) + pflogvars).sum(
                1)
            
            s = s_
            states[:, self.trajectory_length - i - 1] = s
        return states, logpf, logpb, logf

    def sample(self, batch_size, log_r):
        s = torch.zeros(batch_size, self.dim).to(self.device)
        return self.get_trajectory_fwd(s, None, log_r)[0][:, -1]

    def sleep_phase_sample(self, batch_size, exploration_std):
        s = torch.zeros(batch_size, self.dim).to(self.device)
        return self.get_trajectory_fwd(s, exploration_std, log_r=None)[0][:, -1]

    def forward(self, s, exploration_std=None, log_r=None):
        return self.get_trajectory_fwd(s, exploration_std, log_r)
