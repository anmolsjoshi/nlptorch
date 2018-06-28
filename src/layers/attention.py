import torch as T
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
        x_elementwise_y = T.mul(x, y)
        matrix = T.cat([x, y, x_elementwise_y], 3)
        S = self.wsim(matrix)
        S = S.squeeze()
        return S


class BasicAttention(nn.Module):
    def __init__(self, dropout):
        self.drouput = nn.Dropout(dropout)
        super(BasicAttention, self).__init__()
    
    def forward(self, x, y):
        e = T.bmm(x, y.transpose(1, 2))
        ai = F.softmax(e, dim=-1)
        
        a = T.bmm(ai, y)
        b = T.cat([x, a], dim=-1)
        
        out = self.dropout(b)
        
        return out
    
        
class BiAttention(nn.Module):
    def __init__(self):
        super(BiAttention, self).__init__()
    
    def forward(self, S, x, y):
        q_len = x.shape[1]
        c_len = y.shape[1]

        context2query = T.bmm(F.softmax(S, dim=-1), x)

        b = F.softmax(T.max(S, 2)[0], dim=-1)
        query2context = T.bmm(b.unsqueeze(1), y)
        query2context = query2context.repeat(1, c_len, 1)
        
        contextatten = x.mul(context2query)
        queryatten = y.mul(query2context)
        
        return T.cat([x, context2query, contextatten, queryatten], 2)
    
    
class CoAttention(nn.Module):
    def __init__(self, hidden_size):
        super(CoAttention, self).__init__()
        self.Wqj = nn.Linear(2*hidden_size, 2*hidden_size)
        self.c0 = nn.Parameter(T.rand(2*hidden_size,))
        self.q0 = nn.Parameter(T.rand(2 * hidden_size, ))
    
    def forward(self, q, c):
        b, _, l = q.shape
        qj = F.tanh(self.Wqj(q))
        q_vec = self.c0.unsqueeze(0).expand(b, l).unsqueeze(1)
        q = T.cat([qj, q_vec], dim=2)

        c_vec = self.c0.unsqueeze(0).expand(b, l).unsqueeze(1)
        c = T.cat([c, c_vec], dim=2)

        L = T.bmm(q, c.transpose(1, 2))
        alpha = F.softmax(L, dim=2)
        beta = F.softmax(L, dim=1)

        return alpha, beta
