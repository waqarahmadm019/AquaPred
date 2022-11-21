import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, LeakyReLU
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

# GCN-CNN based model

class GAT_GCN(torch.nn.Module):
    def __init__(self, n_output=1, num_features=66,
                 n_filters=32, embed_dim=128, output_dim=64, dropout=0.3):

        super(GAT_GCN, self).__init__()

        self.n_output = n_output
        self.D1_conv1 = GATConv(num_features, num_features, heads=10)
        self.D1_conv2 = GCNConv(num_features*10, num_features*10*2)
        self.D1_fc_g1 = torch.nn.Linear(num_features*10*2, 1000)
        self.D1_fc_g2 = torch.nn.Linear(1000, output_dim)
        
        
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        
        self.fc1 = nn.Linear(output_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        
        self.out = nn.Linear(16, self.n_output)
        

    def forward(self, data):
        x1, edge_index_1, batch1 = data.x.float(), data.edge_index, data.batch
        x1 = self.D1_conv1(x1, edge_index_1)
        x1 = self.relu(x1)
        x1 = self.D1_conv2(x1, edge_index_1)
        x1 = self.relu(x1)

        x1 = self.relu(self.D1_fc_g1(x1))
        x1 = self.dropout(x1)
        x1 = self.D1_fc_g2(x1)
  
        x1 = self.relu(x1)
        x1 = self.dropout(x1)
        
        x1 = self.fc1(x1)
        x1 = self.relu(x1)
        x1 = self.dropout(x1)
        x1 = self.fc2(x1)
        x1 = self.relu(x1)
        x1 = self.dropout(x1)
        
        out = self.out(x1)
        
        return out