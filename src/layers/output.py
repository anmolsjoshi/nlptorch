from torch import nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch as T
from maxout import Maxout


class DynamicPointingDecoder(nn.Module):
    def __init__(self, hidden_dim, pool_size=16, p=0.25, max_iter=4, use_cuda=False):
        super(DynamicPointingDecoder,self).__init__()
        
        self.hidden_dim = hidden_dim
        self.max_iter = max_iter
        self.LSTMdec = nn.LSTMCell(hidden_dim*4,hidden_dim)
        self.HMNstart = HighwayMaxoutNetwork(hidden_dim, pool_size)
        self.HMNend = HighwayMaxoutNetwork(hidden_dim, pool_size)
        self.dropout = nn.Dropout(p)
        self.use_cuda = use_cuda
        
        if use_cuda:
            self.cuda()
        
    def init_hidden(self, batch_size):
        hidden = Variable(T.zeros(batch_size, self.hidden_dim))
        context = Variable(T.zeros(batch_size, self.hidden_dim))
        if self.use_cuda:
            hidden=hidden.cuda()
            context=context.cuda()
        return (hidden,context)
        
    def forward(self, U, is_training=False):
        
        batch_size = U.size(0)
        hidden = self.init_hidden(batch_size)
        si, ei = 0, 1
        us, ue = U[:, si, :], U[:, ei, :]
        entropies=[]
        
        for i in range(self.max_iter):
            entropy, alphas, betas = [], [], []
            
            for ut in U.transpose(0,1):
                alphas.append(self.HMNstart(ut, us, ue, hidden[0]))
            alpha = T.cat(alphas, dim=-1)
            entropy.append(alpha)
            alpha = alpha.max(1)[1]
            us = T.cat([U[i,alpha[i],:].unsqueeze(0) for i in range(batch_size)])
            
            for ut in U.transpose(0,1):
                betas.append(self.HMNend(ut, us, ue, hidden[0]))
            beta = T.cat(betas, dim=-1)
            entropy.append(beta)
            beta = beta.max(1)[1]
            ue = T.cat([U[i,beta[i],:].unsqueeze(0) for i in range(batch_size)])
            
            hidden = self.LSTMdec(T.cat([us, ue], 1), hidden)
            
            si_est = alpha.argmax().item()
            ei_est = beta.argmax().item()
            
            if is_training == False and si == si_est and ei == ei_est:
                entropies.append(entropy)
                break
            else:
                entropies.append(entropy)
                si = si_est
                ei = ei_est
            
        return alpha, beta, entropies