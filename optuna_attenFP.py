#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn import Linear, ReLU, LeakyReLU, BatchNorm1d
from torch_geometric.nn import AttentiveFP #, global_add_pool as add,GlobalAttention, GraphNorm
# from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_sort_pool as sortpool
# from optuna.trial import TrialState
# from torch.nn.modules import activation
# GINConv model
class AttentionConvNet(torch.nn.Module):
    def __init__(self, trial, in_features=92, out_features=91, n_output=1,
                 dropout=0.3):

        super(AttentionConvNet, self).__init__()
        self.trial = trial
        # activation_candidates = {
        #     "relu": torch.nn.ReLU(),
        #     "celu": torch.nn.CELU(),
        #     # "sigmoid": torch.nn.Sigmoid(),
        #     "tanh": torch.nn.Tanh(),
        #     "reakly_relu": torch.nn.LeakyReLU(),
        #     "gelu": torch.nn.GELU(),
        #     # "hardswitch": torch.nn.Hardswish()
        # }
        
        
        
        # nonlinearity_name = trial.suggest_categorical("nonlinearity", list(activation_candidates))
        # nonlinearity = activation_candidates[nonlinearity_name]
        
        
        # pooling_candidate = activation_candidates[pooling_name]
        # dim1 = 100
        # dim2 = 200
        # self.graphlayers = nn.ModuleList()
        # graph_layers = trial.suggest_int("graph_layers", 1, 10)
        # ****start of graph convolution layer 1.****************************************************************
        self.hidden_features1 = self.trial.suggest_int("hidden_features", 2, in_features)
        self.hidden_features = self.trial.suggest_int("out_features", 4, in_features)
        self.dropout = self.trial.suggest_float("dropout", 0.1, 0.7)
        self.num_layers = self.trial.suggest_int("num_layers", 1, 20)
        self.num_timesteps = trial.suggest_int("num_timesteps", 2, 30)
        self.conv1 = AttentiveFP(in_channels=in_features, hidden_channels=self.hidden_features,
                                out_channels=1,edge_dim=10, 
                                num_layers=self.num_layers, num_timesteps=self.num_timesteps,
                                dropout=self.dropout)
        
        self.conv2 = AttentiveFP(in_channels=in_features, hidden_channels=self.hidden_features,
                                out_channels=1,edge_dim=10, 
                                num_layers=self.num_layers, num_timesteps=self.num_timesteps,
                                dropout=self.dropout)
        # self.bn2 = GraphNorm(self.out_features1)
        # self.bn2 = torch.nn.BatchNorm1d(self.out_features1)
        
        
        self.relu = nn.ReLU()
        self.n_output = n_output
        
        
       # pooling_candidates = {
       #      "sum_pool": add,
       #      "mean_pool": gap,
       #      "sort_pool": sortpool,
       #      "max_pool": gmp,
       #      "attention_pool": GlobalAttention(gate_nn = torch.nn.Sequential(torch.nn.Linear(self.out_features1, 2*(self.out_features1)),
       #                                                                torch.nn.BatchNorm1d(2*(self.out_features1)),
       #                                                                torch.nn.ReLU(), torch.nn.Linear(2*(self.out_features1), 1)))
            
       #  }
        # pooling_name = trial.suggest_categorical("pooling_name", list(pooling_candidates))
        # self.pool = activation_candidates[pooling_name]
        # self.pool = sortpool(self.out_features1)
        # self.fc1 = nn.Linear(32, 16)
        # self.fc2 = nn.Linear(16, 8)
        # self.fc3 = nn.Linear(8, 4)
        # self.out = nn.Linear(self.out_features1*15, self.n_output) 
        # self.out = nn.Linear(1, self.n_output)        # n_output = 1 for regression task
    

    def forward(self, data1):
        x, edge_index, batch,edge_attr = data1.x, data1.edge_index, data1.batch,data1.edge_attr
        # x = self.conv1(x, edge_index= edge_index_1, edge_attr=edge_attr)
        # x = self.bn1(x)
        # x = self.relu(x)
        
        # x = self.conv2(x, edge_index= edge_index_1, edge_attr=edge_attr)
        # x = self.bn2(x)
        # x = self.relu(x)
        # num_convlayers = self.trial.suggest_int("num_layers", 2, 10)
        x = self.conv1(x, edge_index, edge_attr, batch)
        # for conv in range(num_convlayers):
        #     # x = conv(x, edge_index= edge_index_1, edge_attr=edge_attr)
        #     x = conv(x, edge_index_1)
        #     x = self.relu(x)
        # x = self.bn2(x)
        # # x = BatchNorm(x)
        # for dense in self.fullylayers:
        #     x = dense(x)
        
        # x = sortpool(x, batch1,15)
        # x = self.pool(x, batch1)
        # x = gap(x, batch1)
        # x = self.fc1(x)
        # x = self.fc2(x)
        # x = self.fc3(x)
        # output = self.out(x)
        return x


