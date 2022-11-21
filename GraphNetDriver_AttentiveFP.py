#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 17:32:52 2022

@author: waqar
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 12:31:49 2022

@author: waqar
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os.path as osp
from math import sqrt
import os
# from torch.autograd import Variable
import torch
import torch.nn.functional as F
from rdkit import Chem
from torchvision.ops import sigmoid_focal_loss 
#from torch_geometric.datasets import MoleculeNet
# from utils import MyOwnDataset #CustomDataset, MoleculeDataset
from torch_geometric.loader import DataLoader
#from torch_geometric.nn.models import AttentiveFP
# from models.gat import GATNet
# from models.attentivefp import AttentiveFP #,MessagePassing
from models.attenFP_v1 import AttentionConvNet
from folds import foldscreator


from rdkit import RDLogger
from tqdm import tqdm
RDLogger.DisableLog('rdApp.*')
from Data_Prep.Graph_Data import Molecule_data
# writer = SummaryWriter('runs/Lipophilacity_experiment_1')
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

print(torch.__version__)
device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
seed = 145
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
the_last_loss = 100
patience = 30
trigger_times = 0
count_loss_difference = 0
#LR = 0.005
# learning_rate = 0.00688267742977242
learning_rate = 0.01688267742977242
weight_decay=0.000307616688331247
#L
#LR = 0.0028894537419258915
LOG_INTERVAL = 20
NUM_EPOCHS = 500
results = []
TRAIN_BATCH_SIZE = 64
NO_OF_FOLDS = 10
weights = [0.23, 0.77]
class_weights = torch.FloatTensor(weights).to(device)
path = osp.join(osp.dirname(osp.realpath(__file__)), 'data')

####reading the file#################################
# df = pd.read_excel('data.xlsx', sheet_name=[0,1])
# traindf = df.get(0)
# testdf = df.get(1)
# df = pd.read_csv('hERG_05_bioactivity_data_2class_pIC50.csv')
# traindf = traindf.sample(frac=1).reset_index(drop=True)
# testdf = testdf.sample(frac=1).reset_index(drop=True)
# traindf = pd.read_csv('New_fold/aqsol_llinas_train.csv')
# testdf = pd.read_csv('New_fold/llinas2020_set1_test.csv')
traindf = pd.read_csv('New_fold/aqsol_llinas_train.csv')
testdf = pd.read_csv('New_fold/llinas2020_set2_test.csv')
# testdf2 = pd.read_csv('New_fold/llinas2020_set2_test.csv')
# traindf.to_csv('New_fold/x_train.csv',index=False)
# testdf.to_csv('New_fold/x_test.csv',index=False)
##############################################################

# processed_data_file_train = 'data/processed/' + 'llinas2020_train_data_set.pt'
# processed_data_file_test = 'data/processed/'  + 'llinas2020_test_data_set1.pt'
processed_data_file_train = 'data/processed/' + 'llinas2020_train_data_set.pt'
processed_data_file_test = 'data/processed/'  + 'llinas2020_test_data_set2.pt'
if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
        print('please run create_data.py to prepare data in pytorch format!')
        foldscreator.createData(traindf, testdf, 'smiles', 'y')





best_ret = []
best_mse = 0.80
best_ci = 0
best_epoch = -1

# for keys in model_loaded.state_dict().keys():
#     print(keys)
train_losses = []
test_losses = []
test_accs = []


def train(model, optimizer,train_loader):
    train_labels = 0
    train_predictions = 0
    total_loss = total_examples = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        # out = model(data)
        y = data.y.view([-1])
        out1 = out.view([-1])
        # print("train : ", y.shape)
        loss = F.mse_loss(out1, y)
        loss.backward()
        optimizer.step()
#         train_labels += train_labels + y
#         train_predictions += train_predictions + out1
        total_loss += float(loss) * data.num_graphs
        total_examples += data.num_graphs
    return total_loss,sqrt(total_loss / total_examples)

@torch.no_grad()
def test(loader, model):
    # mse = []
    model.eval()
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
#         total_preds = torch.cat((total_preds, out1.cpu()), 0)
#         total_labels = torch.cat((total_labels, data.y.view(-1, 1).cpu()), 0)
        # mse.append(test_loss).cpu()
    # return test_loss,float(torch.cat(mse, dim=0).mean().sqrt())
    return total_loss,sqrt(total_loss / total_examples) #,total_labels.numpy().flatten(),total_preds.numpy().flatten()



results = []
best_loss_arr = []
scores = []
true_val = []
pred_val = []

model = AttentionConvNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=weight_decay)

val_losses = []
train_losses = []
mae_arr = []
patience = 60
trigger_times = 0
the_last_loss = 100
best_rmse_arr = []
model_file_name = 'saved_models/llinas2020_test2.model'
result_file_name = 'saved_models/result_llinas2020_test2.csv'

train_data = Molecule_data(root='data', dataset='llinas2020_train_data_set',y=None,smile_graph=None,smiles=None)
test_data = Molecule_data(root='data', dataset='llinas2020_test_data_set2',y=None,smile_graph=None,smiles=None)



train_loader   = DataLoader(train_data,batch_size=TRAIN_BATCH_SIZE,shuffle=True)
test_loader  = DataLoader(test_data,batch_size=TRAIN_BATCH_SIZE,shuffle=True)
best_ret = []

for epoch in range(NUM_EPOCHS):
    train_loss,train_rmse=train(model, optimizer,train_loader)
    test_loss,test_rmse = test(test_loader, model)
#         score = metrics.r2_score(true, prediction)
#         , true, prediction
        
    print(f'Epoch: {epoch:03d}, Loss: {train_rmse:.4f} '
      f'Test: {test_rmse:.4f} ') #f'score: {score:.4f} '   
    
    ret = [epoch,train_rmse,test_rmse]
    
    train_losses.append(train_rmse)
    val_losses.append(test_rmse)
#         scores.append(score)
    # Early Stopping
    the_current_loss = test_rmse   #.item()
    best_ret.append(ret)
    if the_current_loss > the_last_loss:
        trigger_times += 1
        print('trigger times:', trigger_times)

        if trigger_times >= patience:
            print('Early stopping!\nStart to test process.')
            break
    else:
        ret = [epoch,train_rmse,test_rmse] #, score
        trigger_times = 0
        the_last_loss = the_current_loss
        best_rmse = the_current_loss
        
        # torch.save(model.state_dict(), model_file_name)
train_series = pd.Series(train_losses)
val_series = pd.Series(val_losses)
#Creating a dictionary by passing Series objects as values
frame = { 'train': train_series, 'test': val_series }
#Creating DataFrame by passing Dictionary
output = pd.DataFrame(frame)
# print(output)
output.to_csv("losses_test2.csv")
results.append(best_ret)
best_rmse_arr.append(best_rmse)
torch.save(model, model_file_name)

















