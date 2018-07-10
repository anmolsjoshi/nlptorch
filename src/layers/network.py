from torch import nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import torch as T
from src.layers.embedding import CharacterEmbedding, WordEmbedding, CharacterConvEmbedding
from src.layers.highway import Highway
from src.layers.maxout import Maxout
from src.layers.attention import CoAttention


class EmbedText(nn.Module):
    def __init__(self, word_args, char_args, shared):
        super(EmbedText, self).__init__()
        self.shared = shared
        if self.shared:
            self.WordEmbeddingshared = WordEmbedding(**word_args)
            self.CharacterEmbeddingshared = CharacterEmbedding(**char_args)
        else:
            self.WordEmbeddingquestion = WordEmbedding(**word_args)
            self.CharacterEmbeddingquestion = CharacterEmbedding(**char_args)
            self.WordEmbeddingcontext = WordEmbedding(**word_args)
            self.CharacterEmbeddingcontext = CharacterEmbedding(**char_args)
            
    def forward(self, question_words, context_words, question_characters, context_characters):
        
        if self.shared:
            question_words_embed = self.WordEmbeddingshared(question_words)
            question_characters_embed = self.CharacterEmbeddingshared(question_characters)
            context_words_embed = self.WordEmbeddingshared(context_words)
            context_character_embed = self.CharacterEmbeddingshared(context_characters)
        else:
            question_words_embed = self.WordEmbeddingquestion(question_words)
            question_characters_embed = self.CharacterEmbeddingquestion(question_characters)
            context_words_embed = self.WordEmbeddingcontext(context_words)
            context_character_embed = self.CharacterEmbeddingcontext(context_characters)
        
        return question_words_embed, context_words_embed, question_characters_embed, context_character_embed

class Encoder(nn.Module):
    def __init__(self, word_args, char_args, hidden_dim, shared, use_cuda):
        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = char_args['embedding_dim'] + word_args['embedding_dim']
        self.use_cuda = use_cuda
        self.shared = shared
        
        if self.shared:
            self.LSTMshared = nn.LSTM(self.embedding_dim, hidden_dim, bidirectional=True)
            self.CharConvEmbeddingshared = CharacterConvEmbedding(**char_args, use_cuda=self.use_cuda)
        else:
            self.LSTMcontext = nn.LSTM(self.embedding_dim, hidden_dim, bidirectional=True)
            self.LSTMquestion = nn.LSTM(self.embedding_dim, hidden_dim, bidirectional=True)
            self.CharConvEmbeddingcontext = CharacterConvEmbedding(**char_args, use_cuda=self.use_cuda)
            self.CharConvEmbeddingquestion = CharacterConvEmbedding(**char_args, use_cuda=self.use_cuda)
            
        if self.use_cuda:
            self.cuda()
        
    def init_hidden(self, steps):
        hidden = Variable(T.zeros(2, steps, self.hidden_dim))
        context = Variable(T.zeros(2, steps, self.hidden_dim))
        if self.use_cuda:
            hidden=hidden.cuda()
            context=context.cuda()
        return (hidden,context)
    
    def forward(self, question_words, context_words, question_characters, context_characters):
        batch_size, question_len, _ = question_words.shape
        _, context_len, _ = context_words.shape
        
        if self.shared:
            question_charconv = self.CharConvEmbeddingshared(question_characters)
            context_charconv = self.CharConvEmbeddingshared(context_characters)
        else:
            question_charconv = self.CharConvEmbeddingquestion(question_characters)
            context_charconv = self.CharConvEmbeddingcontext(context_characters)
                
        question = T.cat([question_words, question_charconv], dim=-1)
        context = T.cat([context_words, context_charconv], dim=-1)
                
        hidden_question = self.init_hidden(steps=question_len)
        hidden_context = self.init_hidden(steps=context_len)
        
        if self.shared:
            question_encoded, _ = self.LSTMshared(question, hidden_question)
            context_encoded, _ = self.LSTMshared(context, hidden_context)
        else:
            question_encoded, _ = self.LSTMquestion(question, hidden_question)
            context_encoded, _ = self.LSTMcontext(context, hidden_context)
        
        return question_encoded, context_encoded

        
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
        self.WD = nn.Linear(hidden_dim*5, hidden_dim)
        self.W1 = Maxout(hidden_dim*3, hidden_dim, pool_size)
        self.W2 = Maxout(hidden_dim, hidden_dim, pool_size)
        self.W3 = Maxout(hidden_dim*2, 1, pool_size)
        
    def forward(self, ut, us, ue, h):
        r = F.tanh(self.WD(T.cat([h, us, ue], dim=-1)))
        m1t = self.W1(T.cat([ut, r], dim=-1))
        m2t = self.W2(m1t)
        m3t = self.W3(T.cat([m1t, m2t], dim=-1))
        return m3t