import os
import os.path as osp
import pandas as pd
# import numpy as np
# from math import sqrt
# from scipy import stats
from torch_geometric.data import InMemoryDataset, Dataset
from torch_geometric import data as Data
import torch
from rdkit import Chem
#from mol import smiles2graph

import torch
from torch_geometric.data import InMemoryDataset, download_url

# Re,'Ta',Ir','Be' last py dala hy wo check karna hy.
class MyOwnDataset(InMemoryDataset):
    def __init__(self, root, fileType = 'testset_novel.csv', transform=None, pre_transform=None, pre_filter=None):
        
        self.fileType= fileType
        self.symbols = ['K', 'Y', 'V', 'Sm', 'Dy', 'In', 'Lu', 'Hg', 'Co', 'Mg',    #list of all elements in the dataset
            'Cu', 'Rh', 'Hf', 'O', 'As', 'Ge', 'Au', 'Mo', 'Br', 'Ce', 
            'Zr', 'Ag', 'Ba', 'N', 'Cr', 'Sr', 'Fe', 'Gd', 'I', 'Al', 
            'B', 'Se', 'Pr', 'Te', 'Cd', 'Pd', 'Si', 'Zn', 'Pb', 'Sn', 
            'Cl', 'Mn', 'Cs', 'Na', 'S', 'Ti', 'Ni', 'Ru', 'Ca', 'Nd', 
            'W', 'H', 'Li', 'Sb', 'Bi', 'La', 'Pt', 'Nb', 'P', 'F', 'C',
            'Re','Ta','Ir','Be','Tl']

        self.hybridizations = [
            Chem.rdchem.HybridizationType.S,
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
            'other',
        ]

        self.stereos = [
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOANY,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE,
        ]
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['solubility.csv']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # Download to `self.raw_dir`.
        # download_url(url, self.raw_dir)
        pass

    def process(self):
        # Read data into huge `Data` list.
        # fileType = self.name + '.csv'
        # fileType = 'solubility_1.csv'
        #fileType = 'solubility_1.csv'
        filename = osp.join(self.root, 'raw',self.fileType)
        # filename = osp.join('/home/waqar/solubility/code/Aqueous_Solubility/main/data/solubility/raw',fileType)
        # osp.join(self.raw_dir(), fileType)
        #self.raw_dir() + self.name + ".csv"
        self.data = pd.read_csv(filename)
        print(self.data.columns.tolist())
        # print(self.data[5])
        smiles_list = self.data['SMILES']
        labels_list = self.data['logS']
        data_list =[]
        for smiles,label in zip(smiles_list, labels_list):
            print("smiles: ", smiles)
            mol = Chem.MolFromSmiles(smiles)
    
            xs = []
            for atom in mol.GetAtoms():
                symbol = [0.] * len(self.symbols)
                symbol[self.symbols.index(atom.GetSymbol())] = 1.
                #comment degree from 6 to 8
                degree = [0.] * 8
                degree[atom.GetDegree()] = 1.
                formal_charge = atom.GetFormalCharge()
                radical_electrons = atom.GetNumRadicalElectrons()
                hybridization = [0.] * len(self.hybridizations)
                hybridization[self.hybridizations.index(
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
                stereo[self.stereos.index(bond.GetStereo())] = 1.
    
                edge_attr = torch.tensor(
                    [single, double, triple, aromatic, conjugation, ring] + stereo)
    
                edge_attrs += [edge_attr, edge_attr]
    
            if len(edge_attrs) == 0:
                edge_index = torch.zeros((2, 0), dtype=torch.long)
                edge_attr = torch.zeros((0, 10), dtype=torch.float)
            else:
                edge_index = torch.tensor(edge_indices).t().contiguous()
                edge_attr = torch.stack(edge_attrs, dim=0)
            graph = Data.Data(x=torch.Tensor(features),
                      edge_index=edge_index,
                      edge_attr=edge_attr,
                      y=torch.FloatTensor([label]))
            data_list.append(graph)
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

#     def process(self):
#         fileType = self.name + '.csv'
#         filename = osp.join(self.root, self.name, 'raw',fileType)
#         # filename = osp.join('/home/waqar/solubility/code/Aqueous_Solubility/main/data/solubility/raw',fileType)
#         # osp.join(self.raw_dir(), fileType)
#         #self.raw_dir() + self.name + ".csv"
#         self.data = pd.read_csv(filename)
#         print(self.data.columns.tolist())
#         # print(self.data[5])
#         smiles_list = self.data['SMILES']
#         labels_list = self.data['Solubility']
#         data_list =[]
#         for smiles,label in zip(smiles_list, labels_list):
#             #smiles2graph return dictionary.
#             # graph['edge_index'] = edge_index
#             # graph['edge_feat'] = edge_attr
#             # graph['node_feat'] = x
#             # graph['num_nodes'] = len(x)
#             graph=smiles2graph(smiles)
#             # Data.Data(x=torch.Tensor(features),
#             # edge_index=torch.LongTensor(edge_index).transpose(1, 0),
#             # y=torch.FloatTensor([labels]))
#             smiles_graph = Data.Data(x=graph['node_feat'],
#             edge_index=graph['edge_index'],
#             edge_attr = graph['edge_feat'],
#             y=torch.FloatTensor([label]))
                
#                 # append graph, label and target sequence to data list
#             data_list.append(smiles_graph)

#         if self.pre_filter is not None:
#             data_list = [data for data in data_list if self.pre_filter(data)]

#         if self.pre_transform is not None:
#             data_list = [self.pre_transform(data) for data in data_list]
#         print('Graph construction done. Saving to file.')
#         data, slices = self.collate(data_list)
# #         print(data.shape,slices.shape)
#         # save preprocessed data:
#         torch.save((data, slices), self.processed_paths[0])

