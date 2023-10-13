#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import os
#os.environ['CUDA_VISIBLE_DEVICES'] ='0'
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# In[ ]:


def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)

def embd(pe,embed_dim , device):
    pe = torch.flatten(pe,1)
    #print('pe shape: ',pe.shape)
    vocab_size = pe.shape[-1]
    #print(embed_dim)
    em = nn.Embedding(vocab_size, embed_dim, device = device)
    pe = pe.long() # torch.tensor(pe).to(device).long()
    tensor = em(pe)
    #print('after embedding: ',tensor.shape) 
    #tensor = tensor.permute(0, 2, 1)
    batch_size, x, orig_ch = tensor.shape
    
    channels = int(np.ceil(orig_ch / 2) * 2)
    inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
    
    pos_x = torch.arange(x, device=tensor.device).type(inv_freq.type())
    sin_inp_x = torch.einsum("i,j->ij", pos_x, inv_freq)
    emb_x = get_emb(sin_inp_x)
    emb = torch.zeros((x, channels), device=tensor.device).type(tensor.type())
    emb[:, : channels] = emb_x

    cached_penc = emb[None, :, :orig_ch].repeat(batch_size, 1, 1)
    
    return tensor + cached_penc#.permute(0, 2, 1)


def tskip(features,device, embedding_dim = 4, dim = 8, heads = 2, up=3,mode='simple',newup=False,multiconv = True):
    num = len(features)
    upp=[]
    if newup==False:
        for i in range(num):
            upp.append(up)
    else:
        upp=up
        
    scale = dim ** -0.5
    shapes = []
    new_f = []
    for i,f in enumerate(features):
        for j in range(upp[i]):
            f = nn.MaxPool2d(2)(f)
        shapes.append(f.shape)
        features[i] = embd(f,embedding_dim,device)
        new_f.append(None)
    bc = features[0].shape[0]
    for i in range(num):
        k = nn.Linear(features[i].shape[-1], dim, bias=False, device=device)(features[i])
        k = k.view(bc, heads, -1, int(dim/heads))
        v = nn.Linear(features[i].shape[-1], dim, bias=False, device=device)(features[i])
        v = v.view(bc, heads, -1, int(dim/heads))

        for j in range(num):
            if i==0:
                new_f[j] = nn.Linear(features[j].size(2), dim, bias=False, device=device)(features[j])
                new_f[j] = new_f[j].view(bc, heads, -1, int(dim/heads))
                #new_f[j] = torch.einsum('b n a -> b {} n d'.format(heads), new_f[j]).contiguous()
            new_f[j] = torch.einsum('bhid,bhjd->bhij', new_f[j], k) * scale
            new_f[j] = F.softmax(new_f[j], dim=-1)
            new_f[j] = torch.einsum('bhij,bhjd->bhid', new_f[j], v)
       
    
    for j in range(num):
        if multiconv == True:
            new_f[j] = nn.Conv2d(heads, int(heads/2), kernel_size=3, padding=1, device=device)(new_f[j]) ##b 1 n dim/heads
            #print(new_f[j].shape)
            new_f[j] = nn.ReLU()(new_f[j])
            #print(new_f[j].shape)
            if heads%4 ==0:
                new_f[j] = nn.Conv2d(int(heads/2), int(heads/4), kernel_size=3, padding=1, device=device)(new_f[j]) ##b 1 n dim/heads
                #print(new_f[j].shape)
                new_f[j] = nn.ReLU()(new_f[j])
            elif heads ==6:
                new_f[j] = nn.Conv2d(int(heads/2), int(heads/heads), kernel_size=3, padding=1, device=device)(new_f[j]) ##b 1 n dim/heads
                #print(new_f[j].shape)
                new_f[j] = nn.ReLU()(new_f[j])
        else:
            new_f[j] = nn.Conv2d(heads, int(1), kernel_size=3, padding=1, device=device)(new_f[j]) ##b 1 n dim/heads
            new_f[j] = nn.ReLU()(new_f[j])
            
        #print(new_f[j].shape)
        new_f[j] = nn.MaxPool2d((1,int(dim/heads)))(new_f[j]) ##b 1 n 1
        #print(new_f[j].shape)
        new_f[j] = torch.squeeze(new_f[j], dim = (1,3)) ##b n
        #print(new_f[j].shape)
        new_f[j] = new_f[j].view(-1, shapes[j][-3], shapes[j][-2], shapes[j][-1]) ##b c h w
        #print(new_f[j].shape)
        
        
    if mode!='simple':
        for i in range(num):
            upp[i] = upp[i]+1
            
    for j in range(num):
        for i in range(upp[j]):
            new_f[j] = nn.Upsample(scale_factor=2)(new_f[j])
        
    return new_f
        

