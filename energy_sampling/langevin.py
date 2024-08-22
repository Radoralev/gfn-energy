import torch
import numpy as np
from scipy.stats import skew, kurtosis
from energies.utils import torsions_to_conformations, conformations_to_torsions

def adjust_ld_step(current_ld_step, current_acceptance_rate, target_acceptance_rate=0.574, adjustment_factor=0.1):
    """
    Adjust the Langevin dynamics step size based on the current acceptance rate.
    
    :param current_ld_step: Current Langevin dynamics step size.
    :param current_acceptance_rate: Current observed acceptance rate.
    :param target_acceptance_rate: Target acceptance rate, default is 0.574.
    :param adjustment_factor: Factor to adjust the ld_step.
    :return: Adjusted Langevin dynamics step size.
    """
    if current_acceptance_rate > target_acceptance_rate:
        return current_ld_step + adjustment_factor * current_ld_step
    else:
        return current_ld_step - adjustment_factor * current_ld_step

def langevin_dynamics(x, energy, device, args):
    accepted_samples = []
    accepted_logr = []
    acceptance_rate_lst = []
    log_r_original = energy.log_reward(x)
    acceptance_count = 0
    acceptance_rate = 0
    total_proposals = 0
   #log_reward = lambda x: log_reward_original(x)#*627.503
    for i in range(args.max_iter_ls):
        x_conf = torsions_to_conformations(x, energy.tas, energy.rd_conf, device).reshape(x.shape[0], -1)

        x = x.requires_grad_(True)
        log_r_original = energy.log_reward(x)
        r_grad_original = torch.tensor(energy.force(x)).reshape(x.shape[0], -1).to(device)
        if args.ld_schedule:
            ld_step = args.ld_step if i == 0 else adjust_ld_step(ld_step, acceptance_rate, target_acceptance_rate=args.target_acceptance_rate)
        else:
            ld_step = args.ld_step

        new_x = x_conf + ld_step * r_grad_original.detach() + np.sqrt(2 * ld_step) * torch.randn_like(x_conf, device=device)
        #TODO: Here we go from conformations to torsions to conformations to energy. This is not ideal.
        new_x_tas = conformations_to_torsions(new_x.reshape(-1, energy.atom_nr, 3), energy.tas, energy.rd_conf)
        log_r_new = energy.log_reward(new_x_tas)
        r_grad_new = torch.tensor(energy.force(new_x_tas)).reshape(x.shape[0], -1).to(device)

        log_q_fwd = -(torch.norm(new_x - x_conf - ld_step * r_grad_original, p=2, dim=1) ** 2) / (4 * ld_step)
        log_q_bck = -(torch.norm(x_conf - new_x - ld_step * r_grad_new, p=2, dim=1) ** 2) / (4 * ld_step)
        log_accept = (log_r_new.squeeze() - log_r_original.squeeze()) + log_q_bck - log_q_fwd

        accept_mask = torch.rand(x.shape[0], device=device) < torch.exp(torch.clamp(log_accept, max=0))
        acceptance_count += accept_mask.sum().item()
        total_proposals += x.shape[0]

        x = x.detach()
        # After burn-in process
        accepted = new_x[accept_mask].reshape(-1, energy.atom_nr, 3)
        if accept_mask.sum() == 0:
            continue
        accepted = conformations_to_torsions(accepted, energy.tas, energy.rd_conf).to(device)
        if i > args.burn_in:
            accepted_samples.append(accepted)
            accepted_logr.append(log_r_new[accept_mask])
        x[accept_mask] = accepted
        log_r_original[accept_mask] = log_r_new[accept_mask]

        if i % 1 == 0:
            acceptance_rate = acceptance_count / total_proposals
            if i>args.burn_in:
                acceptance_rate_lst.append(acceptance_rate)
            acceptance_count = 0
            total_proposals = 0
            # make a bar that's updated with the acceptance rate
            if i % 25 == 0:
                print(f'Iteration {i}, Acceptance Rate: {acceptance_rate}, LD Step: {ld_step}')
    return torch.cat(accepted_samples, dim=0), torch.cat(accepted_logr, dim=0)