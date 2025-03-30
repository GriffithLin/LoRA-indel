#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/12/12 12:03
# @Author : fhh
# @FileName: CRNN.py
# @Software: PyCharm
import torch
import torch.nn as nn
from models.util import CBAMBlock


class MLTP(nn.Module):
    def __init__(self, vocab_size, embedding_size, output_size, dropout):
        super(MLTP, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.dropout = dropout

        self.embed = torch.nn.Embedding(self.vocab_size, self.embedding_size)

        self.conv1 = torch.nn.Conv1d(in_channels=self.embedding_size,
                                     out_channels=64,
                                     kernel_size=2,
                                     stride=1
                                     )

        self.conv2 = torch.nn.Conv1d(in_channels=self.embedding_size,
                                     out_channels=64,
                                     kernel_size=3,
                                     stride=1
                                     )

        self.conv3 = torch.nn.Conv1d(in_channels=self.embedding_size,
                                     out_channels=64,
                                     kernel_size=4,
                                     stride=1
                                     )

        self.conv4 = torch.nn.Conv1d(in_channels=self.embedding_size,
                                     out_channels=64,
                                     kernel_size=5,
                                     stride=1
                                     )

        self.maxpool1d_1 = torch.nn.MaxPool1d(kernel_size=2, padding=1)
        self.maxpool1d_2 = torch.nn.MaxPool1d(kernel_size=3)
        self.maxpool1d_3 = torch.nn.MaxPool1d(kernel_size=1)
        self.maxpool1d_4 = torch.nn.MaxPool1d(kernel_size=4)

        self.CBAM_1 = CBAMBlock(channel=64, reduction=2, kernel_size=3)
        self.CBAM_2 = CBAMBlock(channel=64, reduction=2, kernel_size=3)
        self.CBAM_3 = CBAMBlock(channel=64, reduction=2, kernel_size=3)
        self.CBAM_4 = CBAMBlock(channel=64, reduction=2, kernel_size=3)
        self.CBAM_5 = CBAMBlock(channel=64, reduction=2, kernel_size=3)

        self.BiRNN1 = nn.GRU(input_size=64,
                             hidden_size=64,
                             num_layers=1,
                             bidirectional=True,
                             batch_first=True
                             )

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
        # x1 = self.maxpool1d(x1)
        # x1.shape torch.Size([batch_size, out_channels, (length-con1d_kernel+1)/kernel])
        rnn_input1 = x1.permute(0, 2, 1)  # [batch_size, seq_len, feature]
        RNN_output1, _ = self.BiRNN1(rnn_input1)
        pool1_input = RNN_output1.permute(0, 2, 1)  # [batch_size, feature, seq_len]
        pool_out = self.maxpool1d_1(pool1_input)
        rc1_out = pool_out.view(pool_out.size(0), -1)

        # RC2
        x2 = self.conv2(x)
        x2 = torch.nn.ReLU()(x2)
        rnn_input2 = x2.permute(0, 2, 1)
        RNN_output2, _ = self.BiRNN1(rnn_input2)
        pool2_input = RNN_output2.permute(0, 2, 1)  # [batch_size, feature, seq_len]
        pool_out = self.maxpool1d_2(pool2_input)
        rc2_out = pool_out.view(pool_out.size(0), -1)

        # RC3
        x3 = self.conv3(x)
        x3 = torch.nn.ReLU()(x3)
        rnn_input3 = x3.permute(0, 2, 1)
        RNN_output3, _ = self.BiRNN1(rnn_input3)
        pool3_input = RNN_output3.permute(0, 2, 1)  # [batch_size, feature, seq_len]
        pool_out = self.maxpool1d_3(pool3_input)
        rc3_out = pool_out.view(pool_out.size(0), -1)

        y = torch.cat([rc1_out, rc2_out, rc3_out], dim=-1)  #
        y = self.dropout(y)

        return y

    def forward(self, cnn_pssm, valid_lens, in_feat=False):
        cnn_pssm2 = self.embed(cnn_pssm)
        # 嵌入层维度 torch.Size([batch, length, embedding_size])

        con1d_input = cnn_pssm2.permute(0, 2, 1)
        # cnn_pssm2 = torch.nn.ReLU()(cnn_pssm2)
        # 通过ReLU实现稀疏后的模型能够更好地挖掘相关特征，拟合训练数据
        cnn_vectors = self.cnn(con1d_input)

        # 全连接层
        out_label = self.out(cnn_vectors)
        return out_label


class MLTP2(nn.Module):
    def __init__(self, vocab_size, embedding_size, output_size, dropout):
        super(MLTP2, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.output_size = output_size
        self.dropout = dropout

        self.embed = torch.nn.Embedding(self.vocab_size, self.embedding_size)

        self.conv1 = torch.nn.Conv1d(in_channels=self.embedding_size,
                                     out_channels=64,
                                     kernel_size=1,
                                     stride=1
                                     )

        self.conv2 = torch.nn.Conv1d(in_channels=64,
                                     out_channels=64,
                                     kernel_size=2,
                                     stride=1
                                     )

        self.conv3 = torch.nn.Conv1d(in_channels=64,
                                     out_channels=64,
                                     kernel_size=3,
                                     stride=1
                                     )

        self.maxpool1d_1 = torch.nn.MaxPool1d(kernel_size=1)
        self.maxpool1d_2 = torch.nn.MaxPool1d(kernel_size=2, padding=1)
        self.maxpool1d_3 = torch.nn.MaxPool1d(kernel_size=3)
        self.maxpool1d_4 = torch.nn.MaxPool1d(kernel_size=4)

        self.BiRNN1 = nn.GRU(input_size=64,
                             hidden_size=64,
                             num_layers=1,
                             bidirectional=False,
                             batch_first=True
                             )

        self.out = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*47, 1000), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1000, 200), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(200, 64), nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, self.output_size)
        )
        self.dropout = torch.nn.Dropout(self.dropout)

    def cnn(self, x):
        # RC1
        x1 = self.conv1(x)
        x1 = torch.nn.ReLU()(x1)
        # x1.shape torch.Size([batch_size, out_channels, length-kernel+1])
        # x1 = self.maxpool1d(x1)
        # x1.shape torch.Size([batch_size, out_channels, (length-con1d_kernel+1)/kernel])
        rnn_input1 = x1.permute(0, 2, 1)  # [batch_size, seq_len, feature]
        RNN_output1, _ = self.BiRNN1(rnn_input1)
        cnn2_input = RNN_output1.permute(0, 2, 1)  # [batch_size, feature, seq_len]
        # pool1_out = self.maxpool1d_1(pool1_input)
        # rc1_out = pool1_out.view(pool1_out.size(0), -1)

        # RC2
        x2 = self.conv2(cnn2_input)
        x2 = torch.nn.ReLU()(x2)
        rnn_input2 = x2.permute(0, 2, 1)
        RNN_output2, _ = self.BiRNN1(rnn_input2)
        cnn3_input = RNN_output2.permute(0, 2, 1)  # [batch_size, feature, seq_len]

        # RC3
        x3 = self.conv3(cnn3_input)
        x3 = torch.nn.ReLU()(x3)
        rnn_input3 = x3.permute(0, 2, 1)
        RNN_output3, _ = self.BiRNN1(rnn_input3)
        pool3_input = RNN_output3.permute(0, 2, 1)  # [batch_size, feature, seq_len]
        # pool_out = self.maxpool1d_3(pool3_input)
        # rc3_out = pool3_input.view(pool3_input.size(0), -1)
        #
        # # rnn_input3 = x3.permute(2, 0, 1)
        # # RNN_output3, _ = self.BiRNN1(rnn_input3)
        # # # RNN_output.shape torch.Tensor(num_steps, batch, 2*num_hiddens)
        # # RNN_output3 = RNN_output3.permute(1, 2, 0)
        # # # RNN_output3.shape torch.Tensor(batch, 2*num_hiddens, num_steps)
        #
        # y = self.dropout(rc3_out)

        return pool3_input

    def forward(self, cnn_pssm, valid_lens, in_feat=False):
        cnn_pssm2 = self.embed(cnn_pssm)
        # 嵌入层维度 torch.Size([batch, length, embedding_size])

        con1d_input = cnn_pssm2.permute(0, 2, 1)
        # cnn_pssm2 = torch.nn.ReLU()(cnn_pssm2)
        # 通过ReLU实现稀疏后的模型能够更好地挖掘相关特征，拟合训练数据
        cnn_vectors = self.cnn(con1d_input)

        # 全连接层
        out_label = self.out(cnn_vectors)
        return out_label
