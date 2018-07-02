from torch import nn as nn
from torch.nn import functional as F

class Highway(nn.Module):

    def __init__(self, input_dim):
        super(Highway, self).__init__()
        self.projection = nn.Linear(input_dim, input_dim)
        self.transform = nn.Linear(input_dim, input_dim)
        self.transform.bias.data.fill_(-2.0)

    def forward(self, x):
        H = F.relu(self.projection(x))
        T = F.sigmoid(self.transform(x))
        y = T * H + (1 - T) * x
        return y