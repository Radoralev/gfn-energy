from torch_geometric.data import Data, Dataset, DataLoader, Batch
from ogb.utils.features import (allowable_features, atom_to_feature_vector,
 bond_to_feature_vector, atom_feature_vector_to_dict, bond_feature_vector_to_dict) 
from rdkit import Chem
import numpy as np
import torch

def smiles2graph(smiles_string):
    """
    Converts SMILES string to graph Data object
    :input: SMILES string (str)
    :return: graph object
    """

    mol = Chem.MolFromSmiles(smiles_string)
    mol = Chem.AddHs(mol)
    # mol = Chem.RemoveHs(mol)
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
        print('Mol has no bonds :()')
        edge_index = np.empty((2, 0), dtype = np.int64)
        edge_attr = np.empty((0, num_bond_features), dtype = np.int64)

    graph = dict()
    graph['edge_index'] = edge_index
    graph['edge_feat'] = edge_attr
    graph['node_feat'] = x
   # graph['pos'] = pos
    graph['num_nodes'] = len(x)
    graph['num_bonds'] = len(mol.GetBonds())

    return graph 

def prep_input(graph, pos=None, device=None):
    datalist = []
    atoms = torch.from_numpy(graph['node_feat']).to(device)
    edge_attr = torch.from_numpy(graph['edge_feat']).to(device)
    
    # # Stack all positions into a single tensor
    pos_tensor = pos
    
    # # Compute pairwise distances in a batched manner
    # dist_matrix = torch.cdist(pos_tensor, pos_tensor, p=2)
    
    # # Apply threshold to get edge indices
    # edge_indices = (dist_matrix < 1.75).nonzero(as_tuple=False)
    
    # # Remove self-loops
    # edge_indices = edge_indices[edge_indices[:, 1] != edge_indices[:, 2]]

    # fully connected graph
    edge_indices = torch.combinations(torch.arange(atoms.size(0)).to(device), with_replacement=False).t().contiguous()
    # Create Data objects
    for i, xyz in enumerate(pos_tensor):
        #edge_index = edge_indices[edge_indices[:, 0] == i][:, 1:].t().contiguous()
        data = Data(atoms=atoms, edge_index=edge_indices, edge_attr=edge_attr, pos=xyz).to(device)
        datalist.append(data)
    
    return datalist