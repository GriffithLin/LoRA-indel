#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
from typing import Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn, Tensor


class FeatureFusionNetwork(nn.Module):

    def __init__(self, d_model=1024, nhead=8, num_featurefusion_layers=1, #d_model��ģ��ά�ȣ�; nhead����ͷע�����е�ͷ����; num_featurefusion_layers�������ںϲ��������
                 dim_feedforward=1024, dropout=0.1, activation="relu"):
        super().__init__()

        featurefusion_layer = FeatureFusionLayer(d_model, nhead, dim_feedforward, dropout, activation)
        self.encoder = Encoder(featurefusion_layer, num_featurefusion_layers)

        decoderCFA_layer = DecoderCFALayer(d_model, nhead, dim_feedforward, dropout, activation)
        decoderCFA_norm = nn.LayerNorm(d_model)
        self.decoder = Decoder(decoderCFA_layer, decoderCFA_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):  #ʹ��Xavier���ȷֲ���ʼ������ά�ȴ���1�Ĳ���
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src1, src2):
        memory_temp, memory_search = self.encoder(src1, src2)
        hs = self.decoder(memory_search, memory_temp)
        return hs
        # return memory_temp, memory_search


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])   #���������ض�ģ��module��N�������deepcopy����������Щ�����洢��һ��nn.ModuleList��
    
    
def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "mish":
        return nn.Mish()
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class FeatureFusionLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=128, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.cross_attn2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear11 = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear12 = nn.Linear(dim_feedforward, d_model)

        self.linear21 = nn.Linear(d_model, dim_feedforward)
        self.dropout2 = nn.Dropout(dropout)
        self.linear22 = nn.Linear(dim_feedforward, d_model)

        self.norm11 = nn.LayerNorm(d_model)
        self.norm12 = nn.LayerNorm(d_model)
        self.norm13 = nn.LayerNorm(d_model)
        self.norm21 = nn.LayerNorm(d_model)
        self.norm22 = nn.LayerNorm(d_model)
        self.norm23 = nn.LayerNorm(d_model)
        self.dropout11 = nn.Dropout(dropout)
        self.dropout12 = nn.Dropout(dropout)
        self.dropout13 = nn.Dropout(dropout)
        self.dropout21 = nn.Dropout(dropout)
        self.dropout22 = nn.Dropout(dropout)
        self.dropout23 = nn.Dropout(dropout)
        
        self.activation1 = _get_activation_fn(activation) #���ݴ�����ַ�������"relu"����ȡ��Ӧ�ļ����
        self.activation2 = _get_activation_fn(activation)  
        
    def forward_post(self, src1, src2):
        # Self-attention for source 1
        src1_attn1, _ = self.self_attn1(src1, src1, src1)
        src1 = src1 + self.dropout11(src1_attn1)
        src1 = self.norm11(src1)
        
        src2_attn1, _ = self.self_attn2(src2, src2, src2)
        src2 = src2 + self.dropout21(src2_attn1)
        src2 = self.norm21(src2)

        # Cross-attention from src1 to src2
        src1_attn2, _ = self.cross_attn1(src1, src2, src2)
        src1 = src1 + self.dropout12(src1_attn2)
        src1 = self.norm12(src1)
        src12 = self.linear12(self.dropout1(self.activation1(self.linear11(src1))))
        src1 = src1 + self.dropout13(src12)
        src1 = self.norm13(src1)
        
        src2_attn2, _ = self.cross_attn2(src2, src1, src1)
        src2 = src2 + self.dropout22(src1_attn2)
        src2 = self.norm22(src2)
        src22 = self.linear22(self.dropout2(self.activation2(self.linear21(src2))))
        src2 = src2 + self.dropout23(src22)
        src2 = self.norm23(src2)       

        return src1, src2
    
    def forward(self, src1, src2):

        return self.forward_post(src1, src2)

            
class Encoder(nn.Module): #������

    def __init__(self, featurefusion_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(featurefusion_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src1, src2):
        output1 = src1
        output2 = src2

        for layer in self.layers: #ÿ��featurefusion_layer�㶼�ᰴ˳����output1��output2����Щ�����������¸�ֵ��output1��output2���Ա�����һ�������д��ݸ���һ��
            output1, output2 = layer(output1, output2)

        return output1, output2
 
 
class DecoderCFALayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=128, dropout=0.1, activation="relu"):
        super().__init__()

        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def forward_post(self, tgt, memory):

        tgt2,_ = self.cross_attn(query=tgt, key=memory, value=memory)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        return tgt

    def forward(self, tgt, memory):

        return self.forward_post(tgt, memory)
        
        

class Decoder(nn.Module):  #���һ��CFA layer

    def __init__(self, decoderCFA_layer, norm=None):
        super().__init__()
        self.layers = _get_clones(decoderCFA_layer, 1)
        self.norm = norm

    def forward(self, tgt, memory):
        output = tgt

        for layer in self.layers: #��ÿ�ε����У����������ῼ��֮ǰ���������Ŀ�������е���һ��λ�ã������Ա�������memory�������µ����
            output = layer(output, memory)

        if self.norm is not None:
            output = self.norm(output)

        return output
