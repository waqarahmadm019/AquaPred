#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 16:14:53 2022

@author: waqar
"""

"""
Optuna example that optimizes multi-layer perceptrons using PyTorch.

In this example, we optimize the validation accuracy of hand-written digit recognition using
PyTorch and FashionMNIST. We optimize the neural network architecture as well as the optimizer
configuration. As it is too time consuming to use the whole FashionMNIST dataset,
we here use a small subset of it.

"""

import os
import os.path as osp
import optuna
from math import sqrt
import math
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
# from torchvision import datasets
# from torchvision import transforms
from optuna_attenFP import AttentionConvNet
# from torch_geometric.datasets import MoleculeNet
# from rdkit import Chem
from torch_geometric.loader import DataLoader
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from utils import MyOwnDataset
device = torch.device('cuda:7' if torch.cuda.is_available() else 'cpu')
BATCHSIZE = 128
CLASSES = 10
DIR = os.getcwd()
EPOCHS = 200
LOG_INTERVAL = 10
N_TRAIN_EXAMPLES = BATCHSIZE * 30
N_VALID_EXAMPLES = BATCHSIZE * 10
early_stop_callback = EarlyStopping(monitor="val_accuracy", min_delta=0.00, patience=16, verbose=False, mode="max")



path = osp.join(osp.dirname(osp.realpath(__file__)), 'data')
dataset = MyOwnDataset(path).shuffle()
# dataset = CustomDataset(path, 'solubility','solubility',5,6).shuffle()
# print(dataset[0])
# dataset = MoleculeNet(path, name='ESOL', pre_transform=GenFeatures()).shuffle()
# dataset = MoleculeNet(path, name='FreeSolv', pre_transform=GenFeatures()).shuffle()
# Epochs = trial.suggest_int("epochs", 200, 2000)
N = len(dataset) // 10
test_dataset = dataset[:N]
# val_dataset = dataset[N:2 * N]
# train_dataset = dataset[2 * N:]
train_dataset = dataset[N:]
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=200, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=True)

# batch = next(iter(test_loader))
# print(batch)

# model = AttentionConvNet(trial).to(device)



def train(model, optimizer):
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
            loss = torch.nan_to_num(loss, nan=torch.finfo(loss.dtype).max)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * data.num_graphs
            total_examples += data.num_graphs
        return total_loss,sqrt(total_loss / total_examples)


@torch.no_grad()
def test(loader, model):
    # mse = []
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
        test_loss = torch.nan_to_num(test_loss, nan=torch.finfo(test_loss.dtype).max) # this will probably crash for non-float types
        # print("no of graphs: ", data.num_graphs)
        total_loss += float(test_loss) * data.num_graphs
        total_examples += data.num_graphs
        # mse.append(test_loss).cpu()
    # return test_loss,float(torch.cat(mse, dim=0).mean().sqrt())
    return total_loss,sqrt(total_loss / total_examples)



    
def objective(trial):

    # Generate the model.
    model = AttentionConvNet(trial).to(device)
    # model = define_model(trial).to(device)
    #,"Adadelta"
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    weightdecay = trial.suggest_float("weightdecay", 10**-5, 10**-2.5, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr,weight_decay=weightdecay)
    # Generate the optimizers.
    

    # Get the FashionMNIST dataset.
    # train_loader, valid_loader = get_mnist()

    # Training of the model.
    # val_losses = []
    # train_losses = []
    
    the_last_loss = 100
    patience = 30
    trigger_times = 0
    count_loss_difference = 0
    # Epochs = trial.suggest_int("epochs", 200, 2000)
    for epoch in range(1, EPOCHS):
        # print("this is model type: ", type(model))
        # print(type(optimizer))
        train_loss,train_rmse = train(model,optimizer)
        # val_loss,val_rmse = test(val_loader, model)
        test_loss,test_rmse = test(test_loader, model)
        # print(f'Epoch: {epoch:03d}, Loss: {train_rmse:.4f} Val: {val_rmse:.4f} '
        #       f'Test: {test_rmse:.4f}')
        print(f'Epoch: {epoch:03d}, Loss: {train_rmse:.4f} '
              f'Test: {test_rmse:.4f}')
        # ret = [epoch,train_loss,test_loss.item()]
        # train_losses.append(train_loss)
        # val_losses.append(test_loss.item())
        
        # Early stopping
        the_current_loss = test_rmse   #.item()
        
        if the_current_loss > the_last_loss:
            trigger_times += 1
            print('trigger times:', trigger_times)
    
            if trigger_times >= patience:
                print('Early stopping!\nStart to test process.')
                break
        else:
            trigger_times = 0
            the_last_loss = the_current_loss
            # torch.save(model.state_dict(), 'Graph_attention_best_01.model')

        # trial.report(train_loss, epoch)
        # trial.report(train_rmse, epoch)
        # trial.report(val_loss, epoch)
        # trial.report(val_rmse, epoch)
        # trial.report(test_loss, epoch)
        #handling exception when saving nan rmse
        # if test_rmse == math.isnan:
        #     test_rmse = torch.nan_to_num(test_rmse, nan=torch.finfo(test_rmse.dtype).max)
        trial.report(test_rmse, epoch)
        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return test_rmse


if __name__ == "__main__":
    study = optuna.create_study(study_name='graph attention1 randomized file', direction="minimize",
                                storage='sqlite:///graph-attention-v1.db',
                                load_if_exists=True)
    study.optimize(objective, n_trials=100000) #, timeout=100000
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    
    print("The end of graph attention study")