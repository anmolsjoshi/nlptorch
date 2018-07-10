import torch as th
from torch import nn as nn
from torch.nn import functional as F

class WordEmbedding(nn.Module):

    def __init__(self, vocab_size, embedding_dim, padding_idx, embeddings=None, trainable=True):
        super(WordEmbedding, self).__init__()
        if embeddings is not None:
            #self.embedding = nn.Embedding.from_pretrained(embeddings=embeddings, freeze=trainable)
            self.embedding = nn.Embedding(embeddings.shape[1], embeddings.shape[1])
            self.embedding.weight = nn.Parameter(th.from_numpy(embeddings))
            self.embedding.requires_grad = trainable
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx)

    def forward(self, x):
        return self.embedding(x)

class CharacterEmbedding(nn.Module):

    def __init__(self, vocab_size, embedding_dim, padding_idx, embeddings=None, trainable=True):
        super(CharacterEmbedding, self).__init__()
        if embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(embeddings=embeddings, freeze=trainable)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx)

    def forward(self, x):
        return self.embedding(x)    
    

class CharacterConvEmbedding(nn.Module):

    def __init__(self, char_embed_dim, embedding_dim, kernel_sizes=[2, 3, 4 ,5], num_filters=50, use_cuda=False):
        super(CharacterConvEmbedding, self).__init__()
        self.conv = nn.ModuleList([nn.Conv2d(1, num_filters, (char_embed_dim, kernel)) for kernel in kernel_sizes])
        self.dropout = nn.Dropout(0.2)
        self.project = nn.Linear(len(kernel_sizes)*num_filters, embedding_dim)
        self.use_cuda = use_cuda
        
        if self.use_cuda:
            self.cuda()

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(-1, x.size(-1), x.size(2)).unsqueeze(1)

        x = [F.relu(conv(x)).squeeze() for conv in self.conv]
        x = [F.max_pool1d(conv, conv.size(2)) for conv in x]
        x = th.cat(x, 1).squeeze()
        x = x.view(batch_size, -1, x.size(1))
        x = F.relu(self.project(x))
        return x