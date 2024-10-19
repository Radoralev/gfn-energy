import argparse
import json
import os
import csv
import torch
import matplotlib.pyplot as plt
import wandb
from tqdm import trange
from buffer import ReplayBuffer
from energies import *
from evaluations import *
from gflownet_losses import *
from langevin import langevin_dynamics
from models import GFN#, EGNNModel, MACEModel, TorchANI_Local
from plot_utils import *
from openmm import unit
# import torchani
from pymbar import other_estimators
from tqdm import tqdm

from mcmc_eval_utils import compute_weights, fed_estimate_Z, calc_ESS

from utils import set_seed, cal_subtb_coef_matrix, fig_to_image, get_gfn_optimizer, get_gfn_forward_loss, \
    get_gfn_backward_loss, get_exploration_std, get_name


parser = argparse.ArgumentParser(description='GFN Linear Regression')
parser.add_argument('--lr_policy', type=float, default=1e-3)
parser.add_argument('--lr_flow', type=float, default=1e-2)
parser.add_argument('--lr_back', type=float, default=1e-3)
parser.add_argument('--model', type=str, default='mlp', choices=['mace', 'mlp', 'egnn', 'attention'])
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--s_emb_dim', type=int, default=64)
parser.add_argument('--t_emb_dim', type=int, default=64)
parser.add_argument('--harmonics_dim', type=int, default=64)
parser.add_argument('--batch_size', type=int, default=300)
parser.add_argument('--epochs', type=int, default=25000)
parser.add_argument('--buffer_size', type=int, default=300 * 1000 * 2)
parser.add_argument('--T', type=int, default=100)
parser.add_argument('--subtb_lambda', type=int, default=2)
parser.add_argument('--t_scale', type=float, default=5.)
parser.add_argument('--log_var_range', type=float, default=4.)
parser.add_argument('--energy', type=str, default='9gmm',
                    choices=('9gmm', '25gmm', 'hard_funnel', 'xtb', 
                             'easy_funnel', 'many_well', 'alanine_vacuum_source', 
                             'alanine_vacuum_target', 'alanine_vacuum_full', 'openmm',
                             'neural', 'nequip'))
parser.add_argument('--mode_fwd', type=str, default="tb", choices=('tb', 'tb-avg', 'db', 'subtb', "pis"))
parser.add_argument('--mode_bwd', type=str, default="tb", choices=('tb', 'tb-avg', 'mle'))
parser.add_argument('--both_ways', action='store_true', default=False)

# For local search
################################################################
parser.add_argument('--local_search', action='store_true', default=False)

# How many iterations to run local search
parser.add_argument('--max_iter_ls', type=int, default=200)

# How many iterations to burn in before making local search
parser.add_argument('--burn_in', type=int, default=100)

# How frequently to make local search
parser.add_argument('--ls_cycle', type=int, default=100)

# langevin step size
parser.add_argument('--ld_step', type=float, default=0.001)

parser.add_argument('--ld_schedule', action='store_true', default=False)

# target acceptance rate
parser.add_argument('--target_acceptance_rate', type=float, default=0.574)


# For replay buffer
################################################################
# high beta give steep priorization in reward prioritized replay sampling
parser.add_argument('--beta', type=float, default=1.)

# low rank_weighted give steep priorization in rank-based replay sampling
parser.add_argument('--rank_weight', type=float, default=1e-2)

# three kinds of replay training: random, reward prioritized, rank-based
parser.add_argument('--prioritized', type=str, default="rank", choices=('none', 'reward', 'rank'))
################################################################
# patience
parser.add_argument('--patience', type=int, default=500)
parser.add_argument('--bwd', action='store_true', default=False)
parser.add_argument('--exploratory', action='store_true', default=False)

parser.add_argument('--sampling', type=str, default="buffer", choices=('sleep_phase', 'energy', 'buffer'))
parser.add_argument('--langevin', action='store_true', default=False)
parser.add_argument('--langevin_scaling_per_dimension', action='store_true', default=False)
parser.add_argument('--conditional_flow_model', action='store_true', default=False)
parser.add_argument('--learn_pb', action='store_true', default=False)
parser.add_argument('--pb_scale_range', type=float, default=0.1)
parser.add_argument('--learned_variance', action='store_true', default=False)
parser.add_argument('--partial_energy', action='store_true', default=False)
parser.add_argument('--exploration_factor', type=float, default=0.1)
parser.add_argument('--exploration_wd', action='store_true', default=False)
parser.add_argument('--clipping', action='store_true', default=False)
parser.add_argument('--lgv_clip', type=float, default=1e2)
parser.add_argument('--gfn_clip', type=float, default=1e4)
parser.add_argument('--zero_init', action='store_true', default=False)
parser.add_argument('--pis_architectures', action='store_true', default=False)
parser.add_argument('--lgv_layers', type=int, default=3)
parser.add_argument('--joint_layers', type=int, default=2)
parser.add_argument('--seed', type=int, default=12345)
parser.add_argument('--weight_decay', type=float, default=1e-7)
parser.add_argument('--use_weight_decay', action='store_true', default=False)
parser.add_argument('--smiles', type=str, default='CCCCCC(=O)OC') # First mol in FreeSolv
parser.add_argument('--temperature', type=int, default=300)
parser.add_argument('--torchani-model', type=str, default='', help="TorchANI model to use")
parser.add_argument('--local_model', type=str, default=None, help="Path to local model")
parser.add_argument('--equivariant_architectures', action='store_true', default=False)
parser.add_argument('--eval', type=int, default=0)
parser.add_argument('--save_eval_samples', action='store_true', default=False)
args = parser.parse_args()

set_seed(args.seed)
if 'SLURM_PROCID' in os.environ:
    args.seed += int(os.environ["SLURM_PROCID"])

eval_data_size = 512
final_eval_data_size = 1024 * 2
plot_data_size = 1024
final_plot_data_size = 2000

if args.pis_architectures:
    args.zero_init = True

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
coeff_matrix = cal_subtb_coef_matrix(args.subtb_lambda, args.T).to(device)

if args.both_ways and args.bwd:
    args.bwd = False

if args.local_search:
    args.both_ways = True

# if args.local_model:
#     args.torchani_model = ''

def get_energy():
    model_args = None
    if args.energy == '9gmm':
        energy = NineGaussianMixture(device=device)
    elif args.energy == '25gmm':
        energy = TwentyFiveGaussianMixture(device=device)
    elif args.energy == 'hard_funnel':
        energy = HardFunnel(device=device)
    elif args.energy == 'easy_funnel':
        energy = EasyFunnel(device=device)
    elif args.energy == 'many_well':
        energy = ManyWell(device=device)
    elif args.energy == 'alanine_vacuum_source':
        energy = Alanine(device=device, phi='source', temp=1000)
        args.smiles = energy.smiles
    elif args.energy == 'alanine_vacuum_target':
        energy = Alanine(device=device, phi='target', temp=1000)
        args.smiles = energy.smiles
    elif args.energy == 'alanine_vacuum_full':
        energy_alanine = Alanine(device=device, phi='full', temp=1000, energy=None)
        if args.local_model:
            if args.local_model.split('/')[-1].startswith('egnn'):
                model, model_args = load_model(model='egnn', filename=args.local_model)
                energy = NeuralEnergy(model=model, smiles=energy_alanine.smiles, batch_size_train=args.batch_size)
            elif args.local_model.split('/')[-1].startswith('mace'):
                model, model_args = load_model(model='mace', filename=args.local_model)
                energy = NeuralEnergy(model=model, smiles=energy_alanine.smiles, batch_size_train=args.batch_size)
        elif 'torchani-' in args.torchani_model:
            model = TorchANI_Local()
            model.load(args.torchani_model)
            energy = TorchANIEnergy(model=model, smiles=energy_alanine.smiles, batch_size=args.batch_size)
        energy = Alanine(device=device, phi='full', temp=1000, energy=energy)
        args.smiles = energy.smiles
    elif args.energy == 'xtb':
        energy = MoleculeFromSMILES_XTB(smiles=args.smiles, temp=args.temperature, solvate=args.solvate)
    elif args.energy == 'openmm':
        energy = OpenMMEnergy(smiles=args.smiles, temp=args.temperature, solvate=args.solvate)
    elif args.energy == 'neural':
        if args.torchani_model == 'ANI-1x_8x':
            model = torchani.models.ANI1x(periodic_table_index=True)
            energy = TorchANIEnergy(model=model, smiles=args.smiles, batch_size=args.batch_size)
        elif args.torchani_model == 'ANI-2x':
            model = torchani.models.ANI2x(periodic_table_index=True)
            energy = TorchANIEnergy(model=model, smiles=args.smiles, batch_size=args.batch_size)
        elif args.torchani_model == 'ANI-1ccx':
            model = torchani.models.ANI1ccx(periodic_table_index=True)
            energy = TorchANIEnergy(model=model, smiles=args.smiles, batch_size=args.batch_size)
        elif 'torchani-' in args.torchani_model:
            model = TorchANI_Local()
            model.load(args.torchani_model)
            energy = TorchANIEnergy(model=model, smiles=args.smiles, batch_size=args.batch_size)
        elif args.local_model:
            if args.local_model.split('/')[-1].startswith('egnn'):
                model, model_args = load_model(model='egnn', filename=args.local_model)
            elif args.local_model.split('/')[-1].startswith('mace'):
                model, model_args = load_model(model='mace', filename=args.local_model)
            energy = NeuralEnergy(model=model, smiles=args.smiles, batch_size_train=args.batch_size)
    print(f'Energy model: {energy}')
    return energy, model_args

def load_model(model, filename):
    with open(filename + '.json', 'r') as f:
        model_args = json.load(f)
        if model == 'egnn':
            model = EGNNModel(in_dim=model_args['in_dim'][0], emb_dim=model_args['emb_dim'], out_dim=model_args['out_dim'], num_layers=model_args['num_layers'], num_atom_features=model_args['in_dim'], equivariant_pred=False)
        elif model == 'mace':
            model = MACEModel(in_dim=model_args['in_dim'], emb_dim=model_args['emb_dim'], out_dim=model_args['out_dim'], num_layers=model_args['num_layers'], equivariant_pred=True)
    model.load_state_dict(torch.load(filename + '.pt'))
    return model, model_args

def eval_step(eval_data, energy, gfn_model, final_eval=False):
    gfn_model.eval()
    metrics = dict()
    log_reward_func = energy.log_reward
    if final_eval:
        calculate_log_Z_statistics(energy, gfn_model, metrics, log_reward_func, num_samples)
    else:
        init_state = torch.zeros(eval_data_size, energy.data_ndim).to(device)
        samples, metrics['eval/log_Z'], metrics['eval/log_Z_lb'], metrics[
            'eval/log_Z_learned'] = log_partition_function(
            init_state, gfn_model, log_reward_func)
        metrics['eval/log_Z'] = add_kBT(metrics['eval/log_Z'])
        metrics['eval/log_Z_lb'] = add_kBT(metrics['eval/log_Z_lb'])
        metrics['eval/log_Z_learned'] = add_kBT(metrics['eval/log_Z_learned'])
    if eval_data is None:
        log_elbo = None
        sample_based_metrics = None
    gfn_model.train()
    return metrics

def calculate_log_Z_statistics(energy, gfn_model, metrics, log_reward_func):
    logZs = []
    logZlbs = []
    logZlearned = []
    for _ in range(2):
        init_state = torch.zeros(final_eval_data_size, energy.data_ndim).to(device)
        samples, log_Z, log_Z_lb, log_Z_learned = log_partition_function(init_state, gfn_model, log_reward_func)
        log_Z = add_kBT(log_Z)
        log_Z_lb = add_kBT(log_Z_lb)
        log_Z_learned = add_kBT(log_Z_learned)
        logZs.append(log_Z.item())
        logZlbs.append(log_Z_lb.item())
        logZlearned.append(log_Z_learned.item())
    metrics['final_eval/mean_log_Z'] = torch.mean(torch.tensor(logZs))
    metrics['final_eval/std_log_Z'] = torch.std(torch.tensor(logZs))
    metrics['final_eval/mean_log_Z_lb'] = torch.mean(torch.tensor(logZlbs))
    metrics['final_eval/std_log_Z_lb'] = torch.std(torch.tensor(logZlbs))
    metrics['final_eval/mean_log_Z_learned'] = torch.mean(torch.tensor(logZlearned))
    metrics['final_eval/std_log_Z_learned'] = torch.std(torch.tensor(logZlearned))

def add_kBT(logz):
    kB = unit.BOLTZMANN_CONSTANT_kB.value_in_unit(unit.hartree/unit.kelvin)
    T = 298.15
    hartree_to_kcal = 627.503
    factor = hartree_to_kcal * kB * T 
    return -logz * factor

def load_gfn(args, energy):
    gfn_model = GFN(
        energy.data_ndim, args.s_emb_dim, args.hidden_dim, args.harmonics_dim, args.t_emb_dim,
        trajectory_length=args.T, clipping=args.clipping, lgv_clip=args.lgv_clip, gfn_clip=args.gfn_clip,
        langevin=args.langevin, learned_variance=args.learned_variance,
        partial_energy=args.partial_energy, log_var_range=args.log_var_range,
        pb_scale_range=args.pb_scale_range, model=args.model, smiles=args.smiles,
        t_scale=args.t_scale, langevin_scaling_per_dimension=args.langevin_scaling_per_dimension,
        conditional_flow_model=args.conditional_flow_model, learn_pb=args.learn_pb,
        pis_architectures=args.pis_architectures, lgv_layers=args.lgv_layers, model_args=None,
        joint_layers=args.joint_layers, zero_init=args.zero_init, device=device,
        equivariant_architectures=args.equivariant_architectures, energy=energy
    ).to(device)
    return gfn_model

def eval():
    # name = get_name(args)
    name_s = f'train_annealed_25k/{args.energy}/{args.smiles}/True/'
    name_v = f'train_annealed_25k/{args.energy}/{args.smiles}/False/'
    # if not args.solvate:
    #     name_load = f'results_pis_architectures/xtb/clipping_lgv_100.0_gfn_10000.0_gfn/fwd/fwd_tb/T_5/tscale_0.25/lvr_4.0/seed_12345/ CCCCCC(=O)OC/2024-08-23_02-14-49/model.pt'
    # else:
    #     name_load = f'results_pis_architectures/xtb/clipping_lgv_100.0_gfn_10000.0_gfn/fwd/fwd_tb/T_5/tscale_0.25/lvr_4.0/seed_12345/ CCCCCC(=O)OC/2024-08-23_02-34-31/model.pt'
    name_load_s = name_s + 'model.pt'
    name_load_v = name_v + 'model.pt'

    print(args.energy)
    T = 298.15
    s_energy = MoleculeFromSMILES_XTB(smiles=args.smiles, temp=T, solvate=True, n_jobs=16)
    v_energy = MoleculeFromSMILES_XTB(smiles=args.smiles, temp=T, solvate=False, n_jobs=16)

    metrics = dict()
    output_file = 'fed_results/eval.csv'

    config = args.__dict__
    config["Experiment"] = "{args.energy}"
    wandb.init(project="GFN Energy Evaluation", config=config, name='eval/' + args.energy + '/' + args.smiles)
    kB = unit.BOLTZMANN_CONSTANT_kB.value_in_unit(unit.hartree/unit.kelvin)
    beta = 1
    hartree_to_kcal = 627.509
    states_list = []
    log_beta_weight_list = []
    log_r_list = []
    log_pfs_list = []
    log_pbs_list = []
    num_samples = args.eval

    gfn_model_s = load_gfn(args, s_energy)
    gfn_model_v = load_gfn(args, v_energy)

    gfn_model_s.load_state_dict(torch.load(f'{name_load_s}'))
    gfn_model_v.load_state_dict(torch.load(f'{name_load_v}'))

    gfn_model_v.eval()
    gfn_model_s.eval()

    v_states_list = []
    s_states_list = []

    log_pfs_v_list = []
    log_pbs_v_list = []

    log_pfs_s_list = []
    log_pbs_s_list = []

    # Generate samples from vacuum model
    print('Generating samples from vacuum model.')
    for _ in range(3):  # Adjust the number of iterations as needed
        initial_state = torch.zeros(num_samples, v_energy.data_ndim).to(device)
        states, log_pfs, log_pbs_, _ = gfn_model_v.get_trajectory_fwd(initial_state, None, v_energy.log_reward)
        v_states_list.append(states[:, -1].cpu().detach())
        log_pfs_v_list.append(log_pfs.cpu().detach())
        log_pbs_v_list.append(log_pbs_.cpu().detach())
    v_states = torch.cat(v_states_list, dim=0)
    log_pfs_v = torch.cat(log_pfs_v_list, dim=0)
    log_pbs_v = torch.cat(log_pbs_v_list, dim=0)

    # Generate samples from solvate model
    print('Generating samples from solvate model.')
    for _ in range(3):  # Adjust the number of iterations as needed
        initial_state = torch.zeros(num_samples, s_energy.data_ndim).to(device)
        states, log_pfs, log_pbs, _ = gfn_model_s.get_trajectory_fwd(initial_state, None, s_energy.log_reward)
        s_states_list.append(states[:, -1].cpu().detach())
        log_pfs_s_list.append(log_pfs.cpu().detach())
        log_pbs_s_list.append(log_pbs.cpu().detach())
    s_states = torch.cat(s_states_list, dim=0)
    log_pfs_s = torch.cat(log_pfs_s_list, dim=0)
    log_pbs_s = torch.cat(log_pbs_s_list, dim=0)
    
    # Compute energies for all combinations
    total_samples = v_states.shape[0]
    vv_energies = torch.zeros(total_samples)
    vs_energies = torch.zeros(total_samples)
    sv_energies = torch.zeros(total_samples)
    ss_energies = torch.zeros(total_samples)

    batch_size = 32  # Adjust batch size as needed
    # Compute energies in batches
    for start in tqdm(range(0, total_samples, batch_size), desc="Computing energies"):
        end = min(start + batch_size, total_samples)
        v_batch = v_states[start:end].to(device)
        s_batch = s_states[start:end].to(device)

        vv_energies[start:end] = v_energy.energy(v_batch).cpu()
        vs_energies[start:end] = v_energy.energy(s_batch).cpu()
        sv_energies[start:end] = s_energy.energy(v_batch).cpu()
        ss_energies[start:end] = s_energy.energy(s_batch).cpu()

    # Convert energies to numpy arrays
    vv_energies_np = vv_energies.numpy()
    vs_energies_np = vs_energies.numpy()
    sv_energies_np = sv_energies.numpy()
    ss_energies_np = ss_energies.numpy()

    # Compute work values
    w_F = (sv_energies_np - vv_energies_np) * beta
    w_R = (vs_energies_np - ss_energies_np) * beta

    # Compute free energy differences using pymbar
    print("Computing free energy differences using pymbar...")

    # EXP estimator
    deltaF_EXP_result = other_estimators.exp(w_F)
    deltaF_EXP = deltaF_EXP_result['Delta_f'] * kB * T * hartree_to_kcal
    deltaF_EXP_std = deltaF_EXP_result['dDelta_f'] * kB * T * hartree_to_kcal
    print(f"EXP estimate: {deltaF_EXP:.4f} ± {deltaF_EXP_std:.4f} kcal/mol")

    # BAR estimator
    deltaF_BAR_result = other_estimators.bar(w_F, w_R)
    deltaF_BAR = deltaF_BAR_result['Delta_f'] * kB * T * hartree_to_kcal
    deltaF_BAR_std = deltaF_BAR_result['dDelta_f'] * kB * T * hartree_to_kcal
    print(f"BAR estimate: {deltaF_BAR:.4f} ± {deltaF_BAR_std:.4f} kcal/mol")


    weights_v = compute_weights(-vv_energies, log_pfs_v, beta).numpy()
    weights_s = compute_weights(-ss_energies, log_pfs_s, beta).numpy()

    ESS_v, ESS_ratio_v = calc_ESS(-vv_energies, log_pfs_v, beta)
    ESS_s, ESS_ratio_s = calc_ESS(-ss_energies, log_pfs_s, beta)

    print(f"ESS vacuum: {ESS_v:.4f} ({ESS_ratio_v:.4f})")
    print(f"ESS solvate: {ESS_s:.4f} ({ESS_ratio_s:.4f})")

    # print('Weights shapes:', weights_v.shape, weights_s.shape, vv_energies_np.shape, ss_energies_np.shape, w_F.shape, w_R.shape)
    from scipy.special import logsumexp

    def weighted_EXP(energies_A, energies_B, weights):
        #check if numpy
        if isinstance(energies_A, np.ndarray):
            energies_A = torch.from_numpy(energies_A)
            energies_B = torch.from_numpy(energies_B)
        exponents = (energies_A - energies_B) # sv is solvation energies on vacuum samples 
        #fix inf 
        exponents = exponents - exponents.max()
        weighted = torch.exp(exponents) * weights
        weighted_sum = weighted.sum()
        return torch.log(weighted_sum)

    # weighted_EXP(sv_energies_gfn, vv_energies_gfn, weights_v)*kB*T*hartree_to_kcal, weighted_EXP(ss_energies_gfn, vs_energies_gfn, weights_s)*kB*T*hartree_to_kcal

    deltaF_weighted_EXP_F = weighted_EXP(sv_energies_np, vv_energies_np, weights_v).item() *kB*T*hartree_to_kcal
    deltaF_weighted_EXP_R = weighted_EXP(ss_energies_np, vs_energies_np, weights_s).item() *kB*T*hartree_to_kcal
    print(f"Weighted EXP estimate (forward): {deltaF_weighted_EXP_F:.4f} kcal/mol")
    print(f"Weighted EXP estimate (reverse): {deltaF_weighted_EXP_R:.4f} kcal/mol")

    # FED_Z estimator
    # delta_f_gfn = fed_estimate_Z(vv_energies_np, ss_energies_np, beta_applied=True)
    # print(f"FED_Z estimate: {delta_f_gfn:.4f} kcal/mol")

    log_weights_v = -vv_energies - log_pfs_v.sum(dim=-1) + log_pbs_v.sum(dim=-1)
    log_weights_s = -ss_energies - log_pfs_s.sum(dim=-1) + log_pbs_s.sum(dim=-1)

    deltaF_GFN = -(logmeanexp(log_weights_s) - logmeanexp(log_weights_v)) * kB * T * hartree_to_kcal
    deltaFlb_GFN = -(log_weights_s.mean() - log_weights_v.mean()) * kB * T * hartree_to_kcal

    print(f"FED_Z estimate: {deltaF_GFN:.4f} kcal/mol")
    print(f"FED_Z lower bound: {deltaFlb_GFN:.4f} kcal/mol")
          
    # create a dictionary to store the results
    results = {
        'deltaF_EXP': deltaF_EXP,
        'deltaF_EXP_std': deltaF_EXP_std,
        'deltaF_BAR': deltaF_BAR,
        'deltaF_BAR_std': deltaF_BAR_std,
        'deltaF_weighted_EXP_F': deltaF_weighted_EXP_F,
        'deltaF_weighted_EXP_R': deltaF_weighted_EXP_R,
        'delta_f_kcal': deltaF_GFN,
        'delta_f_lb_kcal': deltaFlb_GFN,
        'ESS_v': ESS_v,
        'ESS_ratio_v': ESS_ratio_v,
        'ESS_s': ESS_s,
        'ESS_ratio_s': ESS_ratio_s
    }

    # check if .csv file exists if it doesn't create it and write the header
    if not os.path.exists(output_file):
        with open(output_file, 'w', newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(['SMILES', 'deltaF_EXP', 'deltaF_EXP_std', 'deltaF_BAR', 'deltaF_BAR_std', 'deltaF_weighted_EXP_F', 'deltaF_weighted_EXP_R', 'delta_f_GFN','delta_Flb_GFN', 'ESS_v', 'ESS_ratio_v', 'ESS_s', 'ESS_ratio_s'])

    # write as a line to csv file (which could exist already)
    with open(output_file, 'a', newline='') as outfile:
        writer = csv.writer(outfile)
        # round up everything to 4th digit 
        deltaF_EXP = round(deltaF_EXP, 4)
        numeric = [deltaF_EXP, deltaF_EXP_std, deltaF_BAR, deltaF_BAR_std, deltaF_weighted_EXP_F, deltaF_weighted_EXP_R, deltaF_GFN.item(), deltaFlb_GFN.item(), ESS_v.item(), ESS_ratio_v.item(), ESS_s.item(), ESS_ratio_s.item()]
        # round up everything to 4th digit
        numeric = [round(num, 4) for num in numeric]
        writer.writerow([args.smiles, *numeric])

    if args.save_eval_samples:
        # Optionally, save the energies and states
        np.save(f'notebooks/data/{args.smiles}/vv_energies.npy', vv_energies_np)
        np.save(f'notebooks/data/{args.smiles}/vs_energies.npy', vs_energies_np)
        np.save(f'notebooks/data/{args.smiles}/sv_energies.npy', sv_energies_np)
        np.save(f'notebooks/data/{args.smiles}/ss_energies.npy', ss_energies_np)
        np.save(f'notebooks/data/{args.smiles}/v_states.npy', v_states.numpy())
        np.save(f'notebooks/data/{args.smiles}/s_states.npy', s_states.numpy())

if __name__ == '__main__':
    if args.eval:
        eval()
