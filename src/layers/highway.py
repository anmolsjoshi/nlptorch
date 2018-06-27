from torch import nn as nn
from torch.nn import functional as F

class HighwayLayer(nn.Module):

    def __init__(self, input_dim):
        super(HighwayLayer, self).__init__()
        self.projection = nn.Linear(input_dim, input_dim)
        self.transform = nn.Linear(input_dim, input_dim)
        self.transform.bias.data.fill_(-2.0)

    def forward(self, x):
        H = F.relu(self.projection(x))
        T = F.sigmoid(self.transform(x))
        y = T * H + (1 - T) * x
        return y


class Highway(nn.Module):

    def __init__(self, input_dim, num_layers=2):
        super(Highway, self).__init__()
        self.num_layers = 2
        self.highway_layer = nn.ModuleList([HighwayLayer(input_dim) for _ in range(num_layers)])

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.highway_layer(x)
        return x

