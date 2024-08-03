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
        edge_index = np.array(edges_list, dtype = np.int64).T

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


def train_model(model_type, in_dim, out_dim, emb_dim, num_layers, lr, epochs, train_data, val_data, device, patience):
    sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

    print(in_dim)
    # Define the model
    if model_type == 'mace':
        model = MACEModel(in_dim=in_dim, out_dim=out_dim, emb_dim=emb_dim, num_layers=num_layers, equivariant_pred=False, batch_norm=False).to(device, dtype=torch.float32)
    elif model_type == 'egnn':
        model = EGNNModel(in_dim=in_dim[0], out_dim=out_dim, emb_dim=emb_dim, num_layers=num_layers, equivariant_pred=False, num_atom_features=in_dim).to(device, dtype=torch.float32)
    else:
        raise ValueError("Invalid model type. Choose either 'mace' or 'egnn'.")
    # print sum params
    print('Parameter number:', sum(p.numel() for p in model.parameters()))
    # Define the optimizer, loss function, and learning rate scheduler, weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    criterion = torch.nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=patience//2, factor=0.5, verbose=True)
    all_losses = []
    best_loss = np.inf
    best_epoch = 0
    best_model_state = None
    patience_counter = 0

    # Setup dataloaders
    train_dataloader = loader.DataLoader(train_data, batch_size=32, shuffle=True)
    val_dataloader = loader.DataLoader(val_data, batch_size=32, shuffle=False)

    for epoch in range(epochs):
        running_loss = 0.0
        with tqdm(train_dataloader, unit="batch") as tepoch:
            for x in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}")
                x = x.to(device)
                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(x)
                loss = criterion(outputs.squeeze(), x.y.to(torch.float32).squeeze())

                # Backward pass and optimize
                loss.backward()
                optimizer.step()

                # Update running loss
                running_loss += loss.item()
                all_losses.append(loss.item())

                tepoch.set_postfix(loss=(loss.item()) * 627.503)
        # Calculate average loss for the epoch
        avg_loss = running_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}, Loss: {avg_loss}")

        # Step the scheduler
        scheduler.step(avg_loss)

        # Check for early stopping
        val_loss = (eval_model(model, val_dataloader, device)) * 627.503
        print(f"Validation Loss: {val_loss}")
        if val_loss < best_loss:
            best_loss = val_loss
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


def eval_model(model, dataloader, device):
    model.eval()
    criterion = torch.nn.MSELoss()
    running_loss = 0.0
    with torch.no_grad():
        for x in dataloader:
            x = x.to(device)
            outputs = model(x)
            loss = criterion(outputs.squeeze(), x.y.to(torch.float32).squeeze())
            running_loss += loss.item()
    return running_loss/len(dataloader)

train_data = []
val_data = []
test_data = []
molecule_list = list(os.listdir(os.path.join(os.getcwd(), '..', 'conformation_sampling', 'conformers')))
# split the molecule list into train,test,val with random 
np.random.seed(42)
np.random.shuffle(molecule_list)
molecule_number = len(molecule_list)
train_molecules = molecule_list[:int(0.8*molecule_number)]
val_molecules = molecule_list[int(0.8*molecule_number):int(0.9*molecule_number)]
test_molecules = molecule_list[int(0.9*molecule_number):]
print(len(train_molecules), len(val_molecules), len(test_molecules))


def extract_mols(name_list):
    data = []
    for i, dir in enumerate(os.listdir(os.path.join(os.getcwd(), '..', 'conformation_sampling', 'conformers'))):
        if dir in name_list:
            solvent_dir = os.path.join(os.getcwd(), '..','conformation_sampling', 'conformers', dir, 'solvation', 'crest_conformers.xyz')
            vacuum_dir = os.path.join(os.getcwd(), '..','conformation_sampling', 'conformers', dir, 'vacuum', 'crest_conformers.xyz')
            if args.solvation:
                solvent_graphs = extract_graphs(solvent_dir)
                data.extend(solvent_graphs)
            else:        
                vacuum_graphs = extract_graphs(vacuum_dir)
                data.extend(vacuum_graphs)
    return data
print('Extracting train data')
train_data = extract_mols(train_molecules)
print('Extracting val data')
val_data = extract_mols(val_molecules)
print('Extracting test data')
test_data = extract_mols(test_molecules)
print('Number of train samples:', len(train_data))
print('Number of val samples:', len(val_data))
print('Number of test samples:', len(test_data))

# find max number of each atom feature in a molecule
max_atom_features = np.zeros(5, dtype=np.int64)
for sample in train_data+val_data+test_data:
    for i in range(5):
        max_atom_features[i] = max(max_atom_features[i], sample.atoms[:, i].max())

print(max_atom_features.tolist())
# calculate mean and variance of .y 
y = min([sample.y for sample in train_data] + [sample.y for sample in val_data] + [sample.y for sample in test_data])
for sample in train_data:
    sample.y = sample.y - y
for sample in val_data:
    sample.y = sample.y - y
for sample in test_data:
    sample.y = sample.y - y


dataloader_test = loader.DataLoader(test_data, batch_size=32, shuffle=True)


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
    device='cuda', 
    patience=50,)

print('MSE on train data:', eval_model(model, dataloader_train, 'cuda') * 627.503)
print('MSE on val data:', eval_model(model, dataloader_test, 'cuda') * 627.503)



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
}
with open(args.output+'.json', 'w') as f:
    json.dump(model_params, f)





