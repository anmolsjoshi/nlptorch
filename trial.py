
# coding: utf-8

# In[ ]:


from src.data.data_batcher import get_batch_generator
from src.data.vocab import get_glove
import os
import re


# In[ ]:


train_context_path = os.path.join("data", "train.context")
train_qn_path = os.path.join("data", "train.question")
train_ans_path = os.path.join("data", "train.span")
dev_context_path = os.path.join("data", "dev.context")
dev_qn_path = os.path.join("data", "dev.question")
dev_ans_path = os.path.join("data", "dev.span")


# In[ ]:


glove_path = os.path.join("data", "glove.6B.{}d.txt".format(100))

emb_matrix, word2id, id2word = get_glove(glove_path, 100)


# In[ ]:


_PAD = '<pad>'
_UNK = '<unk>'
characters = 'abcdefghijklmnopqrstuvwxyz0123456789,.<>:;{}[]\|=+_-()*&?^%$#@!`~\n/\"\'\\'
characters = list([_PAD, _UNK]) + list(characters)
char2id = {character: i for i, character in enumerate(characters)}


# In[ ]:


klm = get_batch_generator(word2id, char2id, train_context_path, 
                          train_qn_path, train_ans_path, 
                          batch_size=32, context_len=625, 
                          question_len=50, character_len=20, discard_long=False)


# In[ ]:


import torch as th
import torch
from torch import optim
from torch import nn as nn
from torch.nn import functional as F
from src.layers.embedding import CharacterConvEmbedding
from src.layers.attention import BasicAttention, BiAttention, CoAttention
from src.layers.network import EmbedText, Encoder, DynamicPointingDecoder


# In[ ]:


char_args = {'vocab_size': 71, 'embedding_dim': 8, 'padding_idx':0}
char_conv_args = {'char_embed_dim':8, 'embedding_dim':100, 'kernel_sizes':[2, 3, 4], 'num_filters':50, 'use_cuda':True}
word_args = {'vocab_size':4e5, 'embedding_dim':100, 'embeddings':emb_matrix, 'trainable':False, 'padding_idx':0}


# In[ ]:


class Model(nn.Module):
    
    def __init__(self, word_args, char_conv_args, shared=True):
        super(Model, self).__init__()
        self.encoder = Encoder(word_args=word_args, char_args=char_conv_args, 
                               hidden_dim=200, shared=shared, use_cuda=True)
        
        self.attention = CoAttention(hidden_dim=200, use_cuda=True)
                
    def forward(self, q_w, c_w, q_c, c_c):
        question_encoded, context_encoded = self.encoder(q_w, c_w, q_c, c_c)
        U = self.attention(question_encoded, context_encoded)
        return U


# In[ ]:


embedtext = EmbedText(word_args, char_args, shared=True)

encoder_net = Model(word_args, char_conv_args, shared=True)
#encoder_net = torch.nn.DataParallel(encoder_net, device_ids=[0, 1])
decoder_net = DynamicPointingDecoder(400, pool_size=16, p=0.25, max_iter=4, use_cuda=True)
#decoder_net = torch.nn.DataParallel(decoder_net, device_ids=[0, 1])


# In[ ]:


LR = 1e-3
loss_fn = nn.CrossEntropyLoss()
enc_optim = optim.Adam(filter(lambda p: p.requires_grad, encoder_net.parameters()), lr=LR, weight_decay=1e-4)
dec_optim = optim.Adam(decoder_net.parameters(),lr=LR, weight_decay=1e-4)


# In[ ]:


import numpy as np

losses_array = []

for i, batch in enumerate(klm):
      
    question_words_embed, context_words_embed, question_characters_embed, context_character_embed = embedtext(
        th.from_numpy(batch.qn_ids).type(torch.LongTensor), 
          th.from_numpy(batch.context_ids).type(torch.LongTensor), 
          th.from_numpy(batch.qn_char_ids).type(torch.LongTensor), 
          th.from_numpy(batch.context_char_ids).type(torch.LongTensor))
    
    question_words = question_words_embed.cuda().type(torch.cuda.FloatTensor)
    context_words = context_words_embed.cuda().type(torch.cuda.FloatTensor)
    question_characters = question_characters_embed.cuda().type(torch.cuda.FloatTensor)
    context_characters = context_character_embed.cuda().type(torch.cuda.FloatTensor)
    
    starts = th.from_numpy(batch.ans_span)[:,0].cuda().type(torch.cuda.LongTensor)
    ends = th.from_numpy(batch.ans_span)[:,1].cuda().type(torch.cuda.LongTensor)
    
    strt = batch.ans_span[:,0]
    end = batch.ans_span[:,1]
    
    u = end[np.where(end > 625)]
    o = strt[np.where(strt > 625)]
    if len(u) != 0:
        print (i, u)
    if len(o) != 0:
        print (i, o)    
    encoder_net.zero_grad()
    decoder_net.zero_grad()
    
    U = encoder_net(question_words, context_words, question_characters, context_characters)
    _, _, entropies = decoder_net(U, is_training=True)
    
    s_ents, e_ents = list(zip(*entropies))
    
    loss_start, loss_end = 0, 0
    for m in range(len(entropies)):
        loss_start += loss_fn(s_ents[m], starts.view(-1))
        loss_end += loss_fn(e_ents[m], ends.view(-1))
        
    loss = loss_start + loss_end
    
    loss.backward()
    enc_optim.step()
    dec_optim.step()
    
    losses_array.append(loss.item())
    print (i, loss.item())
    
np.save('losses.npy', losses_array)

