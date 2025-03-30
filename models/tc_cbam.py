#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2023/3/28 9:07
# @Author : fhh
# @FileName: tc_cbam.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.util import CBAMBlock

torch.manual_seed(20230226)  # 固定随机种子
torch.backends.cudnn.deterministic = True  # 固定GPU运算方式


class TC_CBAM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout):
        super(TC_CBAM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=embedding_dim,
                                              out_channels=n_filters,
                                              kernel_size=fs,
                                              padding='same')
                                    for fs in filter_sizes])
        self.fc = nn.Sequential(nn.Linear(len(filter_sizes) * n_filters * 10, len(filter_sizes) * n_filters), nn.Mish(),
                                nn.Dropout(),
                                nn.Linear(len(filter_sizes) * n_filters, len(filter_sizes) * n_filters // 2), nn.Mish(),
                                nn.Dropout(),
                                nn.Linear(len(filter_sizes) * n_filters // 2, output_dim)
                                )
        self.CBAMs = CBAMBlock(channel=n_filters, reduction=2, kernel_size=1)

        self.dropout = nn.Dropout(dropout)
        self.Mish = nn.Mish()

    def forward(self, data, length=None, encode='sequence'):
        # 对输入数据进行词向量映射
        embedded = self.embedding(data)
        # 进行维度变换
        embedded = embedded.permute(0, 2, 1)
        # 多分枝卷积
        conved = [self.Mish(conv(embedded)) for conv in self.convs]
        # CBAM
        outs = [self.CBAMs(conv) for conv in conved]
        # 多分枝最大池化
        pooled = [F.max_pool1d(conv, math.ceil(conv.shape[2] // 10)) for conv in outs]
        # 多分枝线性展开
        flatten = [pool.view(pool.size(0), -1) for pool in pooled]
        # 将各分支连接在一起
        cat = self.dropout(torch.cat(flatten, dim=1))
        # 输入全连接层，进行回归输出
        return self.fc(cat)
