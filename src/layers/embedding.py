import torch as th
from torch import nn as nn
from torch.nn import functional as F

class WordEmebdding(nn.Module):

    def __init__(self, vocab_size, embedding_dim, padding_idx, embeddings=None, trainable=True):
        super(WordEmebdding, self).__init__()
        if embeddings is not None:
            self.embedding = nn.Embedding.from_pretrained(embeddings=embeddings, freeze=trainable)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx)

    def forward(self, x):
        return self.embedding(x)

class CharacterEmbedding(nn.Module):

    def __init__(self, vocab_size, char_embed_dim, embedding_dim, kernel_sizes=[2, 3, 4 ,5], num_filters=50):
        super(CharacterEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, char_embed_dim)
        self.conv = nn.ModuleList([nn.Conv2d(1, num_filters, (char_embed_dim, kernel)) for kernel in kernel_sizes])
        self.dropout = nn.Dropout(0.2)
        self.project = nn.Linear(len(kernel_sizes)*num_filters, embedding_dim)

    def forward(self, x):
        batch_size = x.size(0)
        x = self.embedding(x)
        x = x.view(-1, x.size(-1), x.size(2)).unsqueeze(1)

        x = [F.relu(conv(x)).squeeze() for conv in self.conv]
        x = [F.max_pool1d(conv, conv.size(2)) for conv in x]
        x = th.cat(x, 1).squeeze()
        x = x.view(batch_size, -1, x.size(1))
        x = F.relu(self.project(x))
        return x

def main():

    vocab_size = 70
    char_embed_dim = 8
    embedding_dim = 100

    x_char = th.zeros([32, 600, 50], dtype=th.long)
    x_word = th.zeros([32, 600], dtype=th.long)
    char = CharacterEmbedding(vocab_size, char_embed_dim, embedding_dim)(x_char)
    word = WordEmebdding(40000, 100, 1)(x_word)
    print (char.shape)
    print (word.shape)
    print (th.cat([word, char], -1).shape)

if __name__ == '__main__':
    main()
