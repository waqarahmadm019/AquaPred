import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout
from torch_geometric.nn import GATv2Conv, global_add_pool


class GAT(torch.nn.Module):
  """Graph Attention Network"""
  def __init__(self, dim_in=92, dim_h=91, dim_out=1, heads=8):
    super().__init__()
    self.gat1 = GATv2Conv(dim_in, dim_h, heads=heads)
    self.gat2 = GATv2Conv(dim_h*heads, dim_out, heads=1)
    self.optimizer = torch.optim.Adam(self.parameters(),
                                      lr=0.005,
                                      weight_decay=5e-4)

  def forward(self, data):
    x, edge_index, batch,edge_attr = data.x, data.edge_index, data.batch,data.edge_attr
    h = F.dropout(x, p=0.6, training=self.training)
    h = self.gat1(x, edge_index)
    h = F.elu(h)
    h = F.dropout(h, p=0.6, training=self.training)
    h = self.gat2(h, edge_index)
    h = global_add_pool(h, batch)
    return h

