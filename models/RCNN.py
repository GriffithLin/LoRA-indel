#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2023/3/17 11:17
# @Author : fhh
# @FileName: RCNN.py
# @Software: PyCharm
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.util import MaskedAttention, CBAMBlock


class R_CNN(nn.Module):
    def __init__(self, vocab_size=21, embedding_size=192, output_size=21, dropout=0.6, max_pool=5):
        super(R_CNN, self).__init__()

        self.embed = nn.Embedding(vocab_size, embedding_size)
        self.rnn = nn.LSTM(input_size=embedding_size,
                           hidden_size=embedding_size,
                           num_layers=2,
                           batch_first=True,
                           bidirectional=True)

        self.conv1 = torch.nn.Conv1d(in_channels=embedding_size,
                                     out_channels=embedding_size,
                                     kernel_size=2,
                                     stride=1
                                     )

        self.conv2 = torch.nn.Conv1d(in_channels=embedding_size,
                                     out_channels=embedding_size,
                                     kernel_size=3,
                                     stride=1
                                     )

        self.conv3 = torch.nn.Conv1d(in_channels=embedding_size,
                                     out_channels=embedding_size,
                                     kernel_size=4,
                                     stride=1
                                     )

        self.conv4 = torch.nn.Conv1d(in_channels=embedding_size,
                                     out_channels=embedding_size,
                                     kernel_size=5,
                                     stride=1
                                     )

        self.MaxPool1d = torch.nn.MaxPool1d(kernel_size=5)

        # self.CBAM_1 = CBAMBlock(channel=64, reduction=2, kernel_size=3)
        # self.CBAM_2 = CBAMBlock(channel=64, reduction=2, kernel_size=3)
        # self.CBAM_3 = CBAMBlock(channel=64, reduction=2, kernel_size=3)
        # self.CBAM_4 = CBAMBlock(channel=64, reduction=2, kernel_size=3)
        # self.CBAM_5 = CBAMBlock(channel=64, reduction=2, kernel_size=3)

        self.out = nn.Sequential(
            nn.Linear(18432, 2000), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2000, 500), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(500, 100), nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(100, self.output_size)
        )
        self.dropout = torch.nn.Dropout(self.dropout)

    def cnn(self, x):
        # RC1
        x1 = self.conv1(x)
        x1 = torch.nn.ReLU()(x1)
        # x1.shape torch.Size([batch_size, out_channels, length-kernel+1])
        # x1.shape torch.Size([batch_size, out_channels, (length-con1d_kernel+1)/kernel])
        pool_out1 = self.MaxPool1d(x1)
        rc1 = pool_out1.view(pool_out1.size(0), -1)

        # RC2
        x2 = self.conv2(x)
        x2 = torch.nn.ReLU()(x2)
        pool_out2 = self.MaxPool1d(x2)
        rc2 = pool_out2.view(pool_out2.size(0), -1)

        # RC3
        x3 = self.conv3(x)
        x3 = torch.nn.ReLU()(x3)
        pool_out3 = self.maxpool1d_3(x3)
        rc3 = pool_out3.view(pool_out3.size(0), -1)

        y = torch.cat([rc1, rc2, rc3], dim=-1)
        y = self.dropout(y)

        return y

    def forward(self, data, valid_lens, in_feat=False):
        # 嵌入层
        rnn_input = self.embed(data)
        # BiLSTM层
        rnn_out = self.rnn(rnn_input)
        # 多尺度卷积层
        conv_input = rnn_out.permute(0, 2, 1)
        cnn_vectors = self.cnn(conv_input)

        # 全连接层
        out_label = self.out(cnn_vectors)
        return out_label


class RCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout):
        super(RCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=embedding_dim,
                                              out_channels=n_filters,
                                              kernel_size=fs,
                                              padding='same')
                                    for fs in filter_sizes])
        self.rnn = nn.LSTM(input_size=embedding_dim,
                           hidden_size=embedding_dim,
                           num_layers=2,
                           batch_first=True,
                           bidirectional=True)
        self.down = nn.Sequential(nn.Linear(embedding_dim * 2, embedding_dim),
                                  nn.Mish(),
                                  nn.Dropout())

        self.fc = nn.Sequential(nn.Linear(len(filter_sizes) * n_filters * 10, len(filter_sizes) * n_filters), nn.Mish(),
                                nn.Dropout(),
                                nn.Linear(len(filter_sizes) * n_filters, len(filter_sizes) * n_filters // 2), nn.Mish(),
                                nn.Dropout(),
                                nn.Linear(len(filter_sizes) * n_filters // 2, output_dim)
                                )
        self.dropout = nn.Dropout(dropout)
        self.Mish = nn.Mish()

    def forward(self, data, length=None, encode='sequence'):
        embedded = self.embedding(data)

        rnn_out, _ = self.rnn(embedded)
        embedded = self.down(rnn_out)

        embedded = embedded.permute(0, 2, 1)

        conved = [self.Mish(conv(embedded)) for conv in self.convs]

        pooled = [F.max_pool1d(conv, math.ceil(conv.shape[2] // 10)) for conv in conved]

        flatten = [pool.view(pool.size(0), -1) for pool in pooled]

        cat = self.dropout(torch.cat(flatten, dim=1))
        return self.fc(cat)
