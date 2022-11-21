import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

# GINConv model
class GINConvNet(torch.nn.Module):
    def __init__(self, n_output=1,num_features=112, n_filters=32, output_dim=64, dropout=0.2):

        super(GINConvNet, self).__init__()

        dim = 32
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.n_output = n_output
        
        D1_nn1 = Sequential(Linear(num_features, num_features), ReLU(), Linear(num_features, num_features))
        self.D1_conv1 = GINConv(D1_nn1)
        self.D1_bn1 = torch.nn.BatchNorm1d(num_features)

        D1_nn2 = Sequential(Linear(num_features, dim), ReLU(), Linear(dim, dim))
        self.D1_conv2 = GINConv(D1_nn2)
        self.D1_bn2 = torch.nn.BatchNorm1d(dim)
        self.D1_fc1_xd = Linear(dim, output_dim)
        
        # combined layers
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 2)
        self.fc5 = nn.Linear(2, n_output)
    

    def forward(self, data):
        x, edge_index, batch = data.x.float(), data.edge_index, data.batch
        
        x = F.relu(self.D1_conv1(x, edge_index))
        x = self.D1_bn1(x)
        x = F.relu(self.D1_conv2(x, edge_index))
        x = self.D1_bn2(x)
        x = global_add_pool(x, batch)
        x = F.relu(self.D1_fc1_xd(x))
        x = F.dropout(x, p=0.2, training=self.training)
        
        # add some dense layers
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        
        return x

    
    
    
