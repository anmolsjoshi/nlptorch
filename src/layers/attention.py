import torch as th
from torch import nn as nn
from torch.nn import functional as F


class Similarity(nn.Module):
    def __init__(self, hidden_size):
        self.wsim = nn.Linear(6*hidden_size, 1, bias=False)
        super(Similarity, self).__init__()
    
    def forward(self, x, y):
        #batch_size, q_len, vec_size = x.shape
        #_, c_len, _ = y.shape
        #x = x.unsqueeze(2).repeat(1, 1, c_len, 1)
        #y = y.unsqueeze(1).repeat(1, q_len, 1, 1)
        x_elementwise_y = th.mul(x, y)
        matrix = th.cat([x, y, x_elementwise_y], 3)
        S = self.wsim(matrix)
        S = S.squeeze()
        return S


class BasicAttention(nn.Module):
    def __init__(self, dropout):
        self.drouput = nn.Dropout(dropout)
        super(BasicAttention, self).__init__()
    
    def forward(self, x, y):
        e = th.bmm(x, y.transpose(1, 2))
        ai = F.softmax(e, dim=-1)
        
        a = th.bmm(ai, y)
        b = th.cat([x, a], dim=-1)
        
        out = self.dropout(b)
        
        return out
    
        
class BiAttention(nn.Module):
    def __init__(self):
        super(BiAttention, self).__init__()
    
    def forward(self, S, x, y):
        _, q_len, _ = x.shape
        _, c_len, _ = y.shape
        
        context2query = th.bmm(F.softmax(S, dim=-1), x)
        
        b = F.softmax(th.max(S, 2)[0], dim=-1)
        query2context = th.bmm(b.unsqueeze(1), y)
        query2context = query2context.repeat(1, c_len, 1)
        
        contextatten = x.mul(context2query)
        queryatten = y.mul(query2context)
        
        return th.cat([x, context2query, contextatten, queryatten], 2)
    
    
class CoAttention(nn.Module):
    def __init__(self):
        super(CoAttention, self).__init__()
    
    def forward(self, x):
        pass