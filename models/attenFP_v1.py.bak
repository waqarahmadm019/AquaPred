#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
# import torch.nn.functional as F
# from torch.nn import Linear, ReLU, LeakyReLU, BatchNorm1d
from torch_geometric.nn import SAGEConv, AttentiveFP #, global_add_pool as add,GlobalAttention, GraphNorm
# from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_sort_pool as sortpool
# from optuna.trial import TrialState
# from torch.nn.modules import activation
# GINConv model
#from attentive_fp_local import AttentiveFP
class AttentionConvNet(torch.nn.Module):
    def __init__(self, in_features=92, out_features=91, n_output=1,
                 dropout=0.256740503759802):
# 0.126740503759802
        super(AttentionConvNet, self).__init__()
        self.sage = SAGEConv(91, 64, True, bias=True)
        #self.GraphNorm = DiffGroupNorm(in_features,2)
        self.linear = nn.Sequential(nn.Linear(in_features,80), nn.ReLU(),nn.Linear(80,72))
        self.dropout = dropout
        self.hidden_features1 = 64
        self.conv1 = AttentiveFP(in_channels=92, hidden_channels=51,
                                out_channels=1,edge_dim=10, 
                                num_layers=1, num_timesteps=7,
                                dropout=self.dropout)
#         self.conv1 = AttentiveFP(in_channels=92, hidden_channels=175,
#                                 out_channels=1,edge_dim=10, 
#                                 num_layers=1, num_timesteps=10,
#                                 dropout=self.dropout)
        
        self.hidden_features2 = 32
        self.conv2 = AttentiveFP(in_channels=92, hidden_channels=51,
                                out_channels=1,edge_dim=10, 
                                num_layers=3, num_timesteps=7,
                                dropout=self.dropout)
        self.hidden_features1 = 32
        self.conv3 = AttentiveFP(in_channels=32, hidden_channels=32,
                                out_channels=16,edge_dim=10, 
                                num_layers=5, num_timesteps=4,
                                dropout=self.dropout)
        
        self.hidden_features2 = 16
        self.conv4 = AttentiveFP(in_channels=16, hidden_channels=16,
                                out_channels=8,edge_dim=10, 
                                num_layers=5, num_timesteps=4,
                                dropout=self.dropout)
        
        # self.bn2 = GraphNorm(self.out_features1)
        # self.bn2 = torch.nn.BatchNorm1d(self.out_features1)
        self.conv5 = AttentiveFP(in_channels=8, hidden_channels=8,
                                out_channels=4,edge_dim=10, 
                                num_layers=5, num_timesteps=4,
                                dropout=self.dropout)
        self.conv6 = AttentiveFP(in_channels=4, hidden_channels=4,
                                out_channels=1,edge_dim=10, 
                                num_layers=5, num_timesteps=4,
                                dropout=self.dropout)
        
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
        # x = self.GraphNorm(x, batch)
        #x = nn.ReLU(self.sage(x, edge_index))
        x = self.conv1(x, edge_index, edge_attr, batch)
        # x = self.conv2(x, edge_index, edge_attr, batch)
        # x = self.conv3(x, edge_index, edge_attr, batch)
        # x = self.conv4(x, edge_index, edge_attr, batch)
        # x = self.conv5(x, edge_index, edge_attr, batch)
        # x = self.conv6(x, edge_index, edge_attr, batch)
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


