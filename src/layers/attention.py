import torch as T
from torch import nn as nn
from torch.nn import functional as F

class BasicAttention(nn.Module):
    def __init__(self, dropout):
        super(BasicAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, q, c):
        e = T.bmm(c, q.transpose(1, 2))
        ai = F.softmax(e, dim=-1)
        
        a = T.bmm(ai, q)
        b = T.cat([c, a], dim=-1)
        out = self.dropout(b)
        
        return out

class BiAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BiAttention, self).__init__()
        self.wsim = nn.Linear(6*hidden_size, 1, bias=False)
    
    def forward(self, q, c):
        q_len = q.shape[1]
        c_len = c.shape[1]

        q_t = q.unsqueeze(1).repeat(1, c_len, 1, 1)
        c_t = c.unsqueeze(2).repeat(1, 1, q_len, 1)
        q_elementwise_c = T.mul(q_t, c_t)
        
        matrix = T.cat([q_t, c_t, q_elementwise_c], 3)
        S = self.wsim(matrix)
        S = S.squeeze()
        
        context2query = T.bmm(F.softmax(S, dim=-1), q)

        b = F.softmax(T.max(S, 2)[0], dim=-1)
        query2context = T.bmm(b.unsqueeze(1), c).repeat(1, c_len, 1)
        
        contextatten = c.mul(context2query)
        queryatten = c.mul(query2context)
        
        return T.cat([c, context2query, contextatten, queryatten], 2)

class CoAttention(nn.Module):
    def __init__(self, hidden_size):
        super(CoAttention, self).__init__()
        self.Wqj = nn.Linear(2*hidden_size, 2*hidden_size)
        self.c0 = nn.Parameter(T.rand(2*hidden_size,))
        self.q0 = nn.Parameter(T.rand(2 * hidden_size, ))
        self.bilstm = nn.LSTM(6*hidden_size, 2*hidden_size, batch_first=True, bidirectional=True)
    
    def forward(self, q, c):
        b, _, l = q.shape
        q_vec = self.q0.unsqueeze(0).expand(b, l).unsqueeze(1)
        qj = T.cat([q, q_vec], dim=1)
        Qj = F.tanh(self.Wqj(qj))
        
        c_vec = self.c0.unsqueeze(0).expand(b, l).unsqueeze(1)
        Dj = T.cat([c, c_vec], dim=1)

        L = T.bmm(Dj, Qj.transpose(1, 2))
        AQ = F.softmax(L, dim=2)
        AD = F.softmax(L.transpose(1, 2), dim=2)
        
        CQ = T.bmm(AQ.transpose(1, 2), Dj)        
        CD = T.bmm(AD.transpose(1, 2), T.cat([Qj, CQ], dim=-1))
        
        U, _ = self.bilstm(T.cat([Dj, CD], dim=-1))
        
        return U[:,:-1,:]
    
def tile(x, dim, num_tile):
    shape = x.shape
    repeat_dim = [1]*(len(shape)+1)
    repeat_dim[dim] = num_tile
    return x.unsqueeze(dim).repeat(*repeat_dim)