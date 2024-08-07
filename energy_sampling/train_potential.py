import argparse
import contextlib
import json
import numpy as np
import os
import re
import subprocess
import sys
import time
import warnings
import torch
from IPython.display import clear_output, display
from tqdm import tqdm
from tqdm.notebook import tqdm
from torch_geometric import loader
from torch_geometric.data import Batch, Data
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from models.egnn import EGNNModel
from models.mace import MACEModel
from itertools import combinations

import matplotlib.pyplot as plt
#torch set float32
torch.set_default_dtype(torch.float32)

parser = argparse.ArgumentParser(description='Train Potential')
parser.add_argument('--num_layers', type=int, default=3, help='Number of layers')
parser.add_argument('--emb_dim', type=int, default=128, help='Embedding size')
parser.add_argument('--solvation', action='store_true', help='Flag to indicate solvation')
parser.add_argument('--output', type=str, default='output.txt', help='Output file string')
parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight decay')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
args = parser.parse_args()



def get_mol_objects(filename):
    mol_list = []
    energy_list = []
   # energies = 
    with open(filename, "r") as f:
        lines = f.readlines()
        file_str  = "".join(lines)
        atom_num = lines[0]
    xyz_list = file_str.split(atom_num)[1:]
    for i in range(len(xyz_list)):
        x = Chem.MolFromXYZBlock(f'{atom_num.strip()}\n' + xyz_list[i])
        Chem.rdDetermineBonds.DetermineConnectivity(x)
        energy = float(xyz_list[i].split('\n')[0].strip())
        energy_list.append(energy)
        mol_list.append(x)
    return mol_list, energy_list

def get_mol_path(mol_idx, solvation=False):
    if solvation:
        return os.path.join(os.getcwd(), 'conformers', f'molecule_{mol_idx}', 'solvation')
    else:
        return os.path.join(os.getcwd(), 'conformers', f'molecule_{mol_idx}', 'vacuum')
get_mol_path(0)


def xyz_mol2graph(xyz_mol):
    mol = xyz_mol
    mol = Chem.AddHs(mol)
    #mol = Chem.RemoveHs(mol)
    pos = mol.GetConformer().GetPositions()
    #print(mol)
    # atoms
    atom_features_list = []

    for atom in mol.GetAtoms():
        atom_features_list.append([
            atom.GetAtomicNum(),
            # int(atom.GetChiralTag()),
            atom.GetTotalDegree(),
            # atom.GetFormalCharge(),
            atom.GetTotalNumHs(),
            atom.GetNumRadicalElectrons(),
            int(atom.GetHybridization()),
            # int(atom.GetIsAromatic()),
            # int(atom.IsInRing())
        ])
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
        edge_index = np.array(list(combinations(list(np.arange(len(x))), r=2))).T
        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = np.array(edge_features_list, dtype = np.int64)

    else:   # mol has no bonds
        print(f'Mol has no bonds :(')
        edge_index = np.empty((2, 0), dtype = np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype = np.int64)

    graph = dict()
    graph['edge_index'] = edge_index
    graph['edge_feat'] = edge_attr
    graph['node_feat'] = x
    graph['pos'] = pos
    graph['num_nodes'] = len(x)
    graph['num_bonds'] = len(mol.GetBonds())

    return graph 


def extract_graphs(filename):
    mol_objects, mol_ens = get_mol_objects(filename)
    datalist = []
    for mol, en in zip(mol_objects, mol_ens):
        graph = xyz_mol2graph(mol)
        if graph['num_bonds'] == 0:
            continue
        data = Data(
            atoms=torch.from_numpy(graph['node_feat']), 
            edge_index=torch.from_numpy(graph['edge_index']), 
            edge_attr=torch.from_numpy(graph['edge_feat']), 
            pos=torch.from_numpy(graph['pos']),
            y=torch.tensor(en))
        data.validate(raise_on_error=True)
        datalist.append(data)
    return datalist    


def update_plot(losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Loss')
    plt.yscale('log')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    clear_output(wait=True)
    display(plt.gcf())
    plt.close()  # Close the figure to prevent it from being displayed again


def train_model(model_type, in_dim, out_dim, emb_dim, num_layers, lr, epochs, train_data, val_data, special_val_data, device, patience,):
    sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

    print(in_dim)
    # Define the model
    if model_type == 'mace':
        model = MACEModel(in_dim=in_dim[0], out_dim=out_dim, emb_dim=emb_dim, num_layers=num_layers, mlp_dim=emb_dim, equivariant_pred=True, batch_norm=False, num_atom_features=in_dim).to(device, dtype=torch.double)
    elif model_type == 'egnn':
        model = EGNNModel(in_dim=in_dim[0], out_dim=out_dim, emb_dim=emb_dim, num_layers=num_layers, equivariant_pred=False, num_atom_features=in_dim).to(device, dtype=torch.float64)
    else:
        raise ValueError("Invalid model type. Choose either 'mace' or 'egnn'.")
    # print sum params
    print('Parameter number:', sum(p.numel() for p in model.parameters()))
    # Define the optimizer, loss function, and learning rate scheduler, weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    criterion = torch.nn.MSELoss(reduction='none')
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1 ** (1 / 9e5))    
    all_losses = []
    best_loss = np.inf
    best_epoch = 0
    best_model_state = None
    patience_counter = 0

    # Setup dataloaders
    train_dataloader = loader.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_dataloader = loader.DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    special_val_dataloader = loader.DataLoader(special_val_data, batch_size=args.batch_size, shuffle=False)
    for epoch in range(epochs):
        running_loss = 0.0
        with tqdm(train_dataloader, unit="batch") as tepoch:
            for x in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}")
                x = x.to(device)
                # Zero the parameter gradients
                num_atoms = (x.atoms[:, 0] >= 0).sum(dim=-1, dtype=torch.float64) 
                optimizer.zero_grad()

                # Forward pass
                outputs = model(x)
                loss = (criterion(outputs.squeeze(), x.y.to(torch.float64).squeeze())/num_atoms.sqrt()).mean()

                # Backward pass and optimize add clip_grad_norm_
                loss.backward()
                optimizer.step()
                scheduler.step()

                # Update running loss
                running_loss += loss.item()
                all_losses.append(loss.item())

                tepoch.set_postfix(loss=(loss.item()))

        # Calculate average loss for the epoch
        avg_loss = running_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss}")
        print(f"Learning rate: {scheduler.get_last_lr()[0]}")
        # Step the scheduler
        special_preds = []
        for x in special_val_dataloader:
            pred = model(x.to(device))
            special_preds.extend(pred.detach().cpu().tolist())
        print(f"Molecule 0 mean pred: {np.mean(rescale(np.array(special_preds)))} kcal/mol, std pred: {np.std(rescale(np.array(special_preds)))} kcal/mol")
        # Check for early stopping
        val_loss_mse, val_loss_mae = (eval_model(model, val_dataloader, device))
        print(f"Validation MAE: {val_loss_mae}")
        print(f"Validation MSE: {val_loss_mse}")
        if val_loss_mae < best_loss:
            best_loss = val_loss_mae
            best_epoch = epoch
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Load the best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Best model from epoch {best_epoch+1} with loss {best_loss}")

    return model, all_losses, train_dataloader



train_data = []
val_data = []
test_data = []
molecule_list = list(os.listdir(os.path.join(os.getcwd(), '..', 'conformation_sampling', 'conformers')))
# split the molecule list into train,test,val with random 
np.random.seed(42)

def extract_mols(name_list):
    data = []
    for i, dir in enumerate(os.listdir(os.path.join(os.getcwd(), '..', 'conformation_sampling', 'conformers'))):
        if dir in name_list:
            solvent_dir = os.path.join(os.getcwd(), '..','conformation_sampling', 'conformers', dir, 'solvation', 'crest_conformers.xyz')
            vacuum_dir = os.path.join(os.getcwd(), '..','conformation_sampling', 'conformers', dir, 'vacuum', 'crest_conformers.xyz')
            if args.solvation:
                solvent_graphs = extract_graphs(solvent_dir)
                if len(solvent_graphs) > 20:
                    data.extend(solvent_graphs)
            else:
                vacuum_graphs = extract_graphs(vacuum_dir)
                if len(vacuum_graphs) > 20:
                    data.extend(vacuum_graphs)
    return data

# data = extract_mols(molecule_list)

np.random.shuffle(molecule_list)
special_val = 'molecule_0'
molecule_number = len(molecule_list)
from sklearn.model_selection import train_test_split

train_molecules, val_molecules = train_test_split(molecule_list, test_size=0.1)
val_molecules, test_molecules = train_test_split(val_molecules, test_size=0.5)

# check if train, test and val are disjoint
assert len(set(train_molecules).intersection(set(val_molecules))) == 0
assert len(set(train_molecules).intersection(set(test_molecules))) == 0
assert len(set(val_molecules).intersection(set(test_molecules))) == 0

if special_val not in train_molecules:
    train_molecules.add(special_val)
if special_val in val_molecules:
    val_molecules.remove(special_val)
if special_val in test_molecules:
    test_molecules.remove(special_val)
print(len(train_molecules), len(val_molecules), len(test_molecules))





print('Extracting train data')
train_data = extract_mols(train_molecules)
print('Extracting val data')
val_data = extract_mols(val_molecules)
special_val_data = extract_mols([special_val])
print('Extracting test data')
test_data = extract_mols(test_molecules)
print('Number of train samples:', len(train_data))
print('Number of val samples:', len(val_data))
print('Number of test samples:', len(test_data))
# special val data target mean and std
print('Special val data target mean and std:', np.mean([sample.y.item() for sample in special_val_data]), np.std([sample.y.item() for sample in special_val_data]))


# # data = train_data+val_data+test_data
# np.random.shuffle(data)
# train_data, val_data = train_test_split(data, test_size=0.1)
# val_data, test_data = train_test_split(val_data, test_size=0.5)

# find max number of each atom feature in a molecule
max_atom_features = np.zeros(5, dtype=np.int64)
atoms = set()
for sample in train_data+val_data+test_data:
    for atom in sample.atoms[:, 0]:
        atoms.add(atom.item())
    for i in range(5):
        max_atom_features[i] = max(max_atom_features[i], sample.atoms[:, i].max())

# check if there any atoms in sample.atoms[:, 0] that are present in test/val data but not in train data
for sample in train_data:
    for atom in sample.atoms[:, 0]:
        if atom.item() not in atoms:
            print('Atom not in train data:', atom)
            break

print(max_atom_features.tolist())


# normalize targets between 0 and 1 
targets = [sample.y.item() for sample in train_data+val_data+test_data]
mean_target = 0#np.mean(targets)
std_target = 1#np.std(targets)
for sample in train_data+val_data+test_data:
    sample.y = (sample.y - mean_target)/std_target

print('Mean target:', mean_target)
print('Std target:', std_target)
def eval_model(model, dataloader, device):
    model.eval()
    criterion1 = torch.nn.MSELoss()
    criterion2 = torch.nn.L1Loss()
    running_loss_mse = 0.0
    running_loss_mae = 0.0
    with torch.no_grad():
        for x in dataloader:
            x = x.to(device)
            outputs = model(x) 
            outputs = rescale(outputs) 
            loss = criterion1(outputs.squeeze(), x.y.to(torch.float32).squeeze())
            loss2 = criterion2(outputs.squeeze(), x.y.to(torch.float32).squeeze())
            running_loss_mse += loss.item()
            running_loss_mae += loss2.item()
    return running_loss_mse/len(dataloader), running_loss_mae/len(dataloader)

def rescale(outputs):
    outputs = outputs*std_target + mean_target
    return outputs


dataloader_test = loader.DataLoader(test_data, batch_size=args.batch_size, shuffle=True)


emb_dim = args.emb_dim
num_layers = args.num_layers
lr = args.lr
epochs=1000

model, losses, dataloader_train = train_model(
    'egnn', 
    in_dim=max_atom_features+1, 
    out_dim=1, 
    emb_dim=emb_dim, 
    num_layers=num_layers, 
    lr=lr, 
    epochs=epochs, 
    train_data=train_data,
    val_data=val_data, 
    special_val_data=special_val_data,
    device='cuda', 
    patience=150,
)
train_mse, train_mae = eval_model(model, dataloader_train, 'cuda')
print('MSE on train data:', train_mse)
print('MAE on train data:', train_mae)

val_mse, val_mae = eval_model(model, loader.DataLoader(val_data, batch_size=args.batch_size, shuffle=True), 'cuda')
print('MSE on val data:', val_mse)
print('MAE on val data:', val_mae)

test_mse, test_mae = eval_model(model, dataloader_test, 'cuda')
print('MSE on test data:', test_mse)
print('MAE on test data:', test_mae)


# save model
torch.save(model.state_dict(), args.output+'.pt')


# save all model parameters in a json in the same folder
model_params = {
    'in_dim': (max_atom_features+1).tolist(),
    'out_dim': 1,
    'emb_dim': emb_dim,
    'num_layers': num_layers,
    'epochs': epochs,
    'solvation': args.solvation,
    'mean_target': mean_target,
    'std_target': std_target,
}
with open(args.output+'.json', 'w') as f:
    json.dump(model_params, f)





