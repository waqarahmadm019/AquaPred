#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Oct 17 15:29:23 2022

@author: waqar
"""
import os.path as osp
import pandas as pd
import numpy as np
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import Draw
import networkx as nx
from torch_geometric.datasets import MoleculeNet
import matplotlib.pyplot as plt
from sklearn import model_selection, preprocessing, metrics, decomposition

import torch
import os
import pandas as pd
import numpy as np
import json,pickle
from collections import OrderedDict
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
import networkx as nx

from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader

from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
import torch
import torch.nn as nn
import Data_Prep.Graph_Data as gd
from Data_Prep.Graph_Data import Molecule_data
from math import sqrt
from models.attenFP_v1 import AttentionConvNet
# from models.gat import GATNet
# test novel dataset on whole trained model.
from sklearn import model_selection, preprocessing, metrics, decomposition
import matplotlib.pyplot as plt

path = osp.join(osp.dirname(osp.realpath(__file__)), 'data')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
####reading the file#################################
# df = pd.read_excel('data.xlsx')
the_last_loss = 100
patience = 30
trigger_times = 0
count_loss_difference = 0
#LR = 0.005
learning_rate = 0.00688267742977242
weight_decay=0.000307616688331247
#LR = 0.0028894537419258915
LOG_INTERVAL = 20
NUM_EPOCHS = 200
results = []
TRAIN_BATCH_SIZE = 64
folds = 10
results = []
best_rmse_arr = []
bestrmsesum = 0
scores = []
true_val = []
pred_val = []
# fig = plt.figure()
# for fold in tqdm(range(folds)):
# val_losses = []
# train_losses = []
# mae_arr = []
# patience = 30
# trigger_times = 0
# the_last_loss = 100
# whole data set as training dataset
def createTestData(path,filename,datasetname):
    
    df_test = pd.read_csv(path + '/' + filename)
#     df_test  = pd.read_csv('New_fold/testset_novel.csv')
    smiles_test = df_test['SMILES']
#         codIds_test = df_test['CODID']
    solubility_test = df_test['logS']
    solubility_test = solubility_test.to_numpy()


#     smile_graph = {}
#     solubility_arr = []
#     smiles_array = []
    smile_graph_test = {}
    solubility_arr_test = []
    smiles_array_test = []

    for i,smile in enumerate(smiles_test):
        g = gd.smile_to_graph(smile)
        if g != None:
            smile_graph_test[smile] = g
            solubility_arr_test.append(smiles_test[i])
            smiles_array_test.append(smile)

#     train_data = Molecule_data(root='data', dataset='train_data_set_fold_'+str(iy),y=band_gap_arr,
#                                smile_graph=smile_graph,smiles=smiles_array)

    noveltest_data = Molecule_data(root='data', dataset=datasetname,y=solubility_test,
                               smile_graph=smile_graph_test,smiles=smiles_array_test)
    return noveltest_data
#     iy+=1

@torch.no_grad()
def predicting(loader, model):
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_loss = total_examples = 0
    for data in loader:
        data = data.to(device)
        out = model(data)
        # out = model(data)
        # mse.append(F.mse_loss(out, data.y, reduction='none').cpu())
        # return float(torch.cat(mse, dim=0).mean().sqrt())
        y = data.y.view([-1])
        out1 = out.view([-1])
        # print("test : ", y.shape)
        test_loss = F.mse_loss(out1, y)
        # print("no of graphs: ", data.num_graphs)
        total_loss += float(test_loss) * data.num_graphs
        total_examples += data.num_graphs
        total_preds = torch.cat((total_preds, out.view(-1, 1).cpu()), 0)
        total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
#         print("total_labels : ", total_labels.shape)
#         print("total_preds : ", total_preds.shape)
        # mse.append(test_loss).cpu()
    # return test_loss,float(torch.cat(mse, dim=0).mean().sqrt())
    return total_loss,sqrt(total_loss / total_examples),total_labels.numpy().flatten(),total_preds.numpy().flatten()

noveltest_data = createTestData('datasets','testset_novel.csv','testset_novel')
# noveltest_data = createTestData('datasets','dls100_test.csv','DLS100')
# noveltest_data = createTestData('datasets','llinas2020_set1_test.csv','llinas2020_set1_test')
# noveltest_data = createTestData('datasets','llinas2020_set1_test.csv','llinas2020_set1_test')
noveltest_loader  = DataLoader(noveltest_data,batch_size=TRAIN_BATCH_SIZE,shuffle=True)
# model = AttentionConvNet().to(device)
model = AttentionConvNet().to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                         weight_decay=weight_decay)

model_file_name = 'wholetrainmodel.model'
checkpoint = torch.load(model_file_name, map_location=torch.device(device))
model.load_state_dict(checkpoint, strict=False)
#     train_data = Molecule_data(root='data', dataset='train_data_set_fold_'+str(fold),y=None,smile_graph=None,smiles=None)
test_loss,test_rmse, true, prediction = predicting(noveltest_loader, model)


score = metrics.r2_score(true, prediction)
# scores.append(score)
print('Test R2: ', score)
print('Test RMSE:', test_rmse)
print('Test Prediction: ', prediction)