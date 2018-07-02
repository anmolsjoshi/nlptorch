from torch import nn as nn
from torch.nn import functional as F
import torch as T
from .highway import Highway
from .maxout import Maxout

class HighwayNetwork(nn.Module):

    def __init__(self, input_dim, num_layers=2):
        super(Highway, self).__init__()
        self.num_layers = 2
        self.highway_layer = nn.ModuleList([Highway(input_dim) for _ in range(num_layers)])

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.highway_layer(x)
        return x

class HighwayMaxoutNetwork(nn.Module):

    def __init__(self, hidden_dim, pool_size=16):
        super(HighwayMaxoutNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.pool_size = pool_size
        self.WD = nn.Linear(10*hidden_dim, 2*hidden_dim)
        self.W1 = Maxout(pool_size)
        self.W2 = Maxout(pool_size)
        self.W3 = Maxout(pool_size)
        
    def forward(self, ut, us, ue, h):
        r = F.tanh(self.WD(T.cat([h, us, ue], dim=-1)))
        m1t = self.W1(T.cat([ut, r]))
        m2t = self.W2(m1t)
        m3t = self.W3(T.cat[m1t, m2t], dim=-1)