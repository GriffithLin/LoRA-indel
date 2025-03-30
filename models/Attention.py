#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2023/3/24 9:32
# @Author : fhh
# @FileName: Attention.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.util import MaskedAttention, transformer_encode

torch.manual_seed(20230226)  # 固定随机种子
torch.backends.cudnn.deterministic = True  # 固定GPU运算方式


class attention(nn.Module):
    def __init__(self, vocab_size=21, embedding_size=95, output_size=21, dropout=0.6, max_pool=5, num_heads=8):
        super(attention, self).__init__()

        self.embed = nn.Embedding(vocab_size, embedding_size)

        self.MaxPool1d = nn.MaxPool1d(kernel_size=max_pool)

        self.attention_encode = MaskedAttention(input_size=embedding_size,
                                                value_size=output_size,
                                                num_hiddens=embedding_size,
                                                dropout=dropout)

        self.AutoEncode = nn.Sequential(nn.Linear(embedding_size, embedding_size // 2),
                                        nn.Dropout(),
                                        nn.Mish(),
                                        nn.Linear(embedding_size // 2, embedding_size),

                                        )
        self.trans = transformer_encode(dropout, embedding_size, num_heads)
        self.out = nn.Sequential(nn.Flatten(),
                                 nn.Linear(embedding_size*50, embedding_size * 25),
                                 nn.Dropout(),
                                 nn.Mish(),
                                 nn.Linear(embedding_size * 25, embedding_size),
                                 nn.Dropout(),
                                 nn.Mish(),
                                 nn.Linear(embedding_size, output_size)
                                 )

    def forward(self, train_data, valid_lens, in_feat=False):
        # attention
        embedded = self.embed(train_data)
        # attention_out, weight = self.attention_encode(embedded, embedded, embedded, valid_lens)
        #
        # # auto-encoder
        # print(attention_out.shape)
        # encoder_out = self.AutoEncode(attention_out)
        for i in range(4):
            embedded = self.trans(embedded)
        out_label = self.out(embedded)

        if in_feat:
            return embedded, out_label
        else:
            return out_label


class Trans_CNN(nn.Module):
    def __init__(self, vocab_size, embedding_size, n_filters, filter_sizes, output_size=21, dropout=0.6, num_heads=4):
        super(Trans_CNN, self).__init__()

        self.embed = nn.Embedding(vocab_size, embedding_size)

        self.convs = nn.ModuleList([nn.Conv1d(in_channels=embedding_size,
                                              out_channels=n_filters,
                                              kernel_size=fs,
                                              padding='same')
                                    for fs in filter_sizes])

        self.trans = transformer_encode(dropout, 50, embedding_size, num_heads)
        self.fc = nn.Sequential(nn.Linear(len(filter_sizes) * n_filters * 10, len(filter_sizes) * n_filters), nn.Mish(),
                                nn.Dropout(),
                                nn.Linear(len(filter_sizes) * n_filters, len(filter_sizes) * n_filters // 2), nn.Mish(),
                                nn.Dropout(),
                                nn.Linear(len(filter_sizes) * n_filters // 2, output_size)
                                )
        self.dropout = nn.Dropout(dropout)
        self.Mish = nn.Mish()

    def forward(self, train_data, valid_lens, in_feat=False):
        # 词向量映射
        embedded = self.embed(train_data)

        # 提取上下文语义信息
        embedded = self.trans(embedded)
        # 进行维度变换
        embedded = embedded.permute(0, 2, 1)

        # 多分枝卷积
        conved = [self.Mish(conv(embedded)) for conv in self.convs]

        # 多分枝最大池化
        pooled = [F.max_pool1d(conv, math.ceil(conv.shape[2] // 10)) for conv in conved]

        # 多分枝线性展开
        flatten = [pool.view(pool.size(0), -1) for pool in pooled]

        # 将各分支连接在一起
        cat = self.dropout(torch.cat(flatten, dim=1))
        out_label = self.fc(cat)

        if in_feat:
            return embedded, out_label
        else:
            return out_label


class CNN_Trans(nn.Module):
    def __init__(self, vocab_size, embedding_size, n_filters, filter_sizes, output_size=21, dropout=0.6, num_heads=4):
        super(CNN_Trans, self).__init__()

        self.embed = nn.Embedding(vocab_size, embedding_size)

        self.convs = nn.ModuleList([nn.Conv1d(in_channels=embedding_size,
                                              out_channels=n_filters,
                                              kernel_size=fs,
                                              padding='same')
                                    for fs in filter_sizes])

        self.trans = transformer_encode(dropout, 10, n_filters, num_heads)
        self.fc = nn.Sequential(nn.Linear(len(filter_sizes) * n_filters * 10, len(filter_sizes) * n_filters), nn.Mish(),
                                nn.Dropout(),
                                nn.Linear(len(filter_sizes) * n_filters, len(filter_sizes) * n_filters // 2), nn.Mish(),
                                nn.Dropout(),
                                nn.Linear(len(filter_sizes) * n_filters // 2, output_size)
                                )
        self.dropout = nn.Dropout(dropout)
        self.Mish = nn.Mish()

    def forward(self, train_data, valid_lens, in_feat=False):
        # 词向量映射
        embedded = self.embed(train_data)

        # 提取上下文语义信息

        # 进行维度变换
        embedded = embedded.permute(0, 2, 1)

        # 多分枝卷积
        conved = [self.Mish(conv(embedded)) for conv in self.convs]

        # 多分枝最大池化
        pooled = [F.max_pool1d(conv, math.ceil(conv.shape[2] // 10)) for conv in conved]

        # trans
        transed = [self.trans(pool.permute(0, 2, 1)) for pool in pooled]
        # 多分枝线性展开
        # flatten = [pool.view(pool.size(0), -1) for pool in pooled]
        flatten = [trans.view(trans.size(0), -1) for trans in transed]

        # 将各分支连接在一起
        cat = self.dropout(torch.cat(flatten, dim=1))
        out_label = self.fc(cat)

        if in_feat:
            return embedded, out_label
        else:
            return out_label
