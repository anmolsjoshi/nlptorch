import torch as T
from torch import nn as nn
from torch.nn import functional as F

class BasicAttention(nn.Module):
    def __init__(self, dropout):
        super(BasicAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, question, context):
        e = T.bmm(context, question.transpose(1, 2))
        attention_distribution = F.softmax(e, dim=-1)
        attention_output = T.bmm(attention_distribution, question)
        blended_rep = T.cat([context, attention_output], dim=-1)
        out = self.dropout(blended_rep)
        
        return out

class BiAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(BiAttention, self).__init__()
        self.wsim = nn.Linear(6*hidden_dim, 1, bias=False)
    
    def forward(self, question, context):
        question_len = question.shape[1]
        context_len = context.shape[1]

        question_tiled = tile(question, dim=1, num_tile=context_len)
        context_tiled = tile(context, dim=2, num_tile=question_len)
        q_elementwise_c = T.mul(question_tiled, context_tiled)
        
        context_question_matrix = T.cat([question_tiled, context_tiled, q_elementwise_c], 3)
        S = self.wsim(context_question_matrix)
        S = S.squeeze()

        alpha = F.softmax(S, dim=-1)
        context2question = T.bmm(alpha, question)

        beta = F.softmax(T.max(S, dim=-1)[0], dim=-1)
        query2context = T.bmm(beta.unsqueeze(1), context).repeat(1, context_len, 1)
        
        attention_context = context.mul(context2question)
        attention_question = context.mul(query2context)
        
        return T.cat([context, attention_context, attention_question], dim=-1)

class CoAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(CoAttention, self).__init__()
        self.Wqj = nn.Linear(2*hidden_dim, 2*hidden_dim)
        self.c0 = nn.Parameter(T.rand(2*hidden_dim,))
        self.q0 = nn.Parameter(T.rand(2 * hidden_dim, ))
        self.bilstm = nn.LSTM(6*hidden_dim, 2*hidden_dim, batch_first=True, bidirectional=True)
    
    def forward(self, question, context):
        b, _, l = question.shape
        question_sentinel = self.q0.unsqueeze(0).expand(b, l).unsqueeze(1)
        question_with_sentinel = T.cat([question, question_sentinel], dim=1)
        Q = F.tanh(self.Wqj(question_with_sentinel))
        
        context_sentinel = self.c0.unsqueeze(0).expand(b, l).unsqueeze(1)
        context_with_sentinel = T.cat([context, context_sentinel], dim=1)

        L = T.bmm(context_with_sentinel, question_with_sentinel.transpose(1, 2))
        attention_question_dist = F.softmax(L, dim=2)
        attention_context_dist = F.softmax(L.transpose(1, 2), dim=2)
        
        attention_question = T.bmm(attention_question_dist.transpose(1, 2), context_with_sentinel)
        attention_context = T.bmm(attention_context_dist.transpose(1, 2),
                                  T.cat([question_with_sentinel, attention_question], dim=-1))
        
        U, _ = self.bilstm(T.cat([context_with_sentinel, attention_context], dim=-1))
        
        return U[:,:-1,:]
    
def tile(x, dim, num_tile):
    shape = x.shape
    repeat_dim = [1]*(len(shape)+1)
    repeat_dim[dim] = num_tile
    return x.unsqueeze(dim).repeat(*repeat_dim)


