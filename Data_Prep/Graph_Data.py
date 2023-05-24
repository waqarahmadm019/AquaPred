import numpy as np
from rdkit import Chem

import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import MolFromSmiles, rdmolops

from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
from scipy.sparse import coo_matrix
import torch
import torch.nn as nn
import networkx as nx
import os

def smiles_features(mol):
    symbols = ['K', 'Y', 'V', 'Sm', 'Dy', 'In', 'Lu', 'Hg', 'Co', 'Mg',    #list of all elements in the dataset
        'Cu', 'Rh', 'Hf', 'O', 'As', 'Ge', 'Au', 'Mo', 'Br', 'Ce', 
        'Zr', 'Ag', 'Ba', 'N', 'Cr', 'Sr', 'Fe', 'Gd', 'I', 'Al', 
        'B', 'Se', 'Pr', 'Te', 'Cd', 'Pd', 'Si', 'Zn', 'Pb', 'Sn', 
        'Cl', 'Mn', 'Cs', 'Na', 'S', 'Ti', 'Ni', 'Ru', 'Ca', 'Nd', 
        'W', 'H', 'Li', 'Sb', 'Bi', 'La', 'Pt', 'Nb', 'P', 'F', 'C',
        'Re','Ta','Ir','Be','Tl']

    hybridizations = [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        'other',
    ]

    stereos = [
        Chem.rdchem.BondStereo.STEREONONE,
        Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ,
        Chem.rdchem.BondStereo.STEREOE,
    ]
    features = []
    xs = []
    adj = rdmolops.GetAdjacencyMatrix(mol, useBO=True)
    for atom in mol.GetAtoms():
        symbol = [0.] * len(symbols)
        symbol[symbols.index(atom.GetSymbol())] = 1.
        #comment degree from 6 to 8
        degree = [0.] * 8
        degree[atom.GetDegree()] = 1.
        formal_charge = atom.GetFormalCharge()
        radical_electrons = atom.GetNumRadicalElectrons()
        hybridization = [0.] * len(hybridizations)
        hybridization[hybridizations.index(
            atom.GetHybridization())] = 1.
        aromaticity = 1. if atom.GetIsAromatic() else 0.
        hydrogens = [0.] * 5
        hydrogens[atom.GetTotalNumHs()] = 1.
        chirality = 1. if atom.HasProp('_ChiralityPossible') else 0.
        chirality_type = [0.] * 2
        if atom.HasProp('_CIPCode'):
            chirality_type[['R', 'S'].index(atom.GetProp('_CIPCode'))] = 1.
    
        x = torch.tensor(symbol + degree + [formal_charge] +
                         [radical_electrons] + hybridization +
                         [aromaticity] + hydrogens + [chirality] +
                         chirality_type)
        xs.append(x)
    
        features = torch.stack(xs, dim=0)

    edge_indices = []
    edge_attrs = []
    for bond in mol.GetBonds():
        edge_indices += [[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]]
        edge_indices += [[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()]]
    
        bond_type = bond.GetBondType()
        single = 1. if bond_type == Chem.rdchem.BondType.SINGLE else 0.
        double = 1. if bond_type == Chem.rdchem.BondType.DOUBLE else 0.
        triple = 1. if bond_type == Chem.rdchem.BondType.TRIPLE else 0.
        aromatic = 1. if bond_type == Chem.rdchem.BondType.AROMATIC else 0.
        conjugation = 1. if bond.GetIsConjugated() else 0.
        ring = 1. if bond.IsInRing() else 0.
        stereo = [0.] * 4
        stereo[stereos.index(bond.GetStereo())] = 1.
    
        edge_attr = torch.tensor(
            [single, double, triple, aromatic, conjugation, ring] + stereo)
    
        edge_attrs += [edge_attr, edge_attr]
    
    if len(edge_attrs) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 10), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_indices).t().contiguous()
        edge_attr = torch.stack(edge_attrs, dim=0)
    return features, edge_index, edge_attr,adj
    

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    if mol == None:
        return None
    
    c_size = mol.GetNumAtoms()
    features, edge_index, edge_attr,adj = smiles_features(mol)
    # features = []
    # bonds = mol.GetBonds()
    # for atom in mol.GetAtoms():
    #     feature = atom_features(atom)
    #     features.append( feature / sum(feature) )

    # edges = []
    # for bond in mol.GetBonds():
    #     edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
    # g = nx.Graph(edges).to_directed()
    # edge_index = []
    # for e1, e2 in g.edges:
    #     edge_index.append([e1, e2])
        
    return c_size, features, edge_index, edge_attr,adj





class Molecule_data(InMemoryDataset):
    def __init__(self, root='/tmp', dataset='davis', y=None, transform=None,
                 pre_transform=None,smile_graph=None,smiles=None):

        #root is required for save preprocessed data, default is '/tmp'
        super(Molecule_data, self).__init__(root, transform, pre_transform)
        # benchmark dataset, default = 'davis'
        self.dataset = dataset
        if os.path.isfile(self.processed_paths[0]):
#             print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(y,smile_graph,smiles)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass
        #return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        # Download to `self.raw_dir`.
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, y,smile_graph,smiles):
       
        data_list = []
        data_len = len(y)
        for i in range(data_len):
            print('Converting SMILES to graph: {}/{}'.format(i+1, data_len))
            smile = smiles[i]
            label = y[i]
            # convert SMILES to molecular representation using rdkit
            c_size, features, edge_index,edge_attr,adj = smile_graph[smile]
            # make the graph ready for PyTorch Geometrics GCN algorithms:
            graph = Data(x=torch.Tensor(features),
                      edge_index=edge_index,
                      edge_attr=edge_attr,
                      y=torch.FloatTensor([label]),
                      A=adj,
                      smiles=str(smile),
                      )
            # GCNData = Data(x=torch.Tensor(features),
            #                     edge_index=torch.LongTensor(edge_index).transpose(1, 0),
            #                     y=torch.FloatTensor([labels]))
            
            
            graph.__setitem__('c_size', torch.LongTensor([c_size]))
            # append graph, label and target sequence to data list
            data_list.append(graph)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        # save preprocessed data:
        torch.save((data, slices), self.processed_paths[0])