import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool as gmp

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


class GCNNet(torch.nn.Module):
    def __init__(self, num_features=112, n_output=1, n_filters=32,output_dim=128, dropout=0.1):
        super(GCNNet, self).__init__()

        self.D1_gcn1 = GCNConv(num_features, 336)
        self.D1_gcn2 = GCNConv(336, 168)
        self.D1_fc_g1 = nn.Linear(168, 84)
        self.fc1 = nn.Linear(84, 42)
        #self.fc2 = nn.Linear(32, 16)
        self.out = nn.Linear(42, 1)

        # activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        x1, edge_index_1, batch1 = data.x.float(), data.edge_index, data.batch
      #  x1 = F.dropout(x1, p=0.2, training=self.training)
        x1 = F.relu(self.D1_gcn1(x1, edge_index_1))
        #x1 = self.relu(x1)
        x1 = F.dropout(x1, p=0.2, training=self.training)
        x1 = self.D1_gcn2(x1, edge_index_1)
        x1 = self.relu(x1)
        x1 = gmp(x1, batch1)          # global max pooling
        x1 = self.D1_fc_g1(x1)
        x1 = self.relu(x1)
        
        # add some dense layers
        x1 = self.fc1(x1)
        x1 = self.relu(x1)
    #    x1 = self.dropout(x1)
   #     x1 = self.fc2(x1)
       # x1 = self.relu(x1)
        #x1 = self.dropout(x1)
        out = self.out(x1)
        return out
    
    
    
class GCNNetTemp(torch.nn.Module):
    def __init__(self, num_features=92, n_output=1, n_filters=32,output_dim=128, dropout=0.1):
        super(GCNNet, self).__init__()

        self.D1_gcn1 = GCNConv(num_features, num_features*8)
        self.D1_gcn2 = GCNConv(num_features * 8, num_features*6)
        self.D1_fc_g1 = nn.Linear(num_features*6, num_features*4)
        self.fc1 = nn.Linear(num_features*4, output_dim)
        self.fc2 = nn.Linear(output_dim, 64)
        self.out = nn.Linear(64, n_output)

        # activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        # graph input feed-forward for drug1
        x1, edge_index_1, batch1 = data.x.float(), data.edge_index, data.batch
      #  x1 = F.dropout(x1, p=0.2, training=self.training)
        x1 = F.elu(self.D1_gcn1(x1, edge_index_1))
        x1 = self.relu(x1)
        x1 = F.dropout(x1, p=0.2, training=self.training)
        x1 = self.D1_gcn2(x1, edge_index_1)
        x1 = self.relu(x1)
        x1 = gmp(x1, batch1)          # global max pooling
        x1 = self.D1_fc_g1(x1)
        x1 = self.relu(x1)
        
        # add some dense layers
        x1 = self.fc1(x1)
        x1 = self.relu(x1)
        x1 = self.dropout(x1)
        x1 = self.fc2(x1)
        x1 = self.relu(x1)
        #x1 = self.dropout(x1)
        out = self.out(x1)
        return out