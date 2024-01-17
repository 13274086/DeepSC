# -*- coding: utf-8 -*-
"""
Created on Mon May 25 20:33:53 2020

@author: HQ Xie
这是一个Transformer的网络结构
"""
"""
Transformer includes:
    Encoder
        1. Positional coding
        2. Multihead-attention
        3. PositionwiseFeedForward
    Decoder
        1. Positional coding
        2. Multihead-attention
        3. Multihead-attention
        4. PositionwiseFeedForward
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import math
from encoder import Encoder
from decoder import Decoder


class ChannelDecoder(nn.Module):
    def __init__(self, in_features, size1, size2):
        super(ChannelDecoder, self).__init__()
        
        self.linear1 = nn.Linear(in_features, size1)
        self.linear2 = nn.Linear(size1, size2)
        self.linear3 = nn.Linear(size2, size1)
        # self.linear4 = nn.Linear(size1, d_model)
        
        self.layernorm = nn.LayerNorm(size1, eps=1e-6)
        
    def forward(self, x):
        x1 = self.linear1(x)
        x2 = F.relu(x1)
        x3 = self.linear2(x2)
        x4 = F.relu(x3)
        x5 = self.linear3(x4)
        
        output = self.layernorm(x1 + x5)

        return output
        
class DeepSC(nn.Module):
    def __init__(self, num_layers, d_model, 
                 num_heads, dff, dropout = 0.1,
                 q=8, v=8, h=8):
        super(DeepSC, self).__init__()
        
        self.encoder = nn.ModuleList([Encoder(d_model,
                                            q,
                                            v,
                                            h,
                                            attention_size=None,
                                            dropout=dropout,
                                            chunk_mode='chunk',
                                            dff=dff) 
                                      for _ in range(num_layers)])
        
        self.channel_encoder = nn.Sequential(nn.Linear(d_model, 256), 
                                             #nn.ELU(inplace=True),
                                             nn.ReLU(inplace=True),
                                             nn.Linear(256, 16))


        self.channel_decoder = ChannelDecoder(16, d_model, 512)
        
        self.decoder = nn.ModuleList([Decoder(d_model,
                                            q,
                                            v,
                                            h,
                                            attention_size=None,
                                            dropout=dropout,
                                            chunk_mode='chunk')
                                      for _ in range(num_layers)])
        
        self.dense = nn.Linear(d_model, trg_vocab_size)
        
        



    
        
        
        
        
        

    

    
    
    
    
    


    


