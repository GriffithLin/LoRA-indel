#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2022/12/20 15:38
# @Author : fhh
# @FileName: T_RNN.py
# @Software: PyCharm
import torch
import torch.nn as nn
from models.util import CBAMBlock, AdditiveAttention


class MLTP3(nn.Module):
    def __init__(self, vocab_size, embedding_size, output_size, dropout):
        super(MLTP3, self).__init__()

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

        self.maxpool1d_1 = torch.nn.MaxPool1d(kernel_size=1)
        self.maxpool1d_2 = torch.nn.MaxPool1d(kernel_size=1)
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
        self.TR = TCN_Block(192)
        self.liner_out = nn.Sequential(
            nn.Linear(18432, 2000), nn.ReLU(),
            nn.Dropout(),
            nn.Linear(2000, 500), nn.ReLU(),
            nn.Dropout(),
            nn.Linear(500, 100), nn.ReLU(),
            nn.Dropout(),
            nn.Linear(100, self.output_size)
        )

        for i in range(21):
            setattr(self, "NaiveFC%d" % i, nn.Sequential(
                nn.Linear(in_features=18432, out_features=128),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(in_features=128, out_features=1),
                nn.Sigmoid()
            ))

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

        cnn_pssm2 = cnn_pssm2.permute(0, 2, 1)
        conv1d_input = torch.nn.ReLU()(cnn_pssm2)
        # 通过ReLU实现稀疏后的模型能够更好地挖掘相关特征，拟合训练数据
        # out = self.TR(con1d_input)
        # print(out.shape)
        cnn_vectors = self.cnn(conv1d_input)

        # 线性label_out
        liner_label = self.liner_out(cnn_vectors)

        # 二分类label_out
        binary_outs = []
        for i in range(21):
            FClayer = getattr(self, "NaiveFC%d" % i)
            y = FClayer(cnn_vectors)
            binary_outs.append(y)
        binary_outs = torch.tensor([i.cpu().detach().numpy() for i in binary_outs], requires_grad=True).cuda()
        binary_outs = torch.squeeze(binary_outs, dim=-1).permute(1, 0)
        return liner_label, binary_outs


class TCN_Block(nn.Module):
    def __init__(self, embedding_size):
        super(TCN_Block, self).__init__()
        self.embedding_size = embedding_size
        self.conv_layer1_1 = torch.nn.Conv1d(in_channels=self.embedding_size,
                                             out_channels=64,
                                             kernel_size=2,
                                             stride=1
                                             )

        self.conv_layer1_2 = torch.nn.Conv1d(in_channels=self.embedding_size,
                                             out_channels=64,
                                             kernel_size=3,
                                             stride=1
                                             )

        self.conv_layer1_3 = torch.nn.Conv1d(in_channels=self.embedding_size,
                                             out_channels=64,
                                             kernel_size=4,
                                             stride=1
                                             )

        self.conv_layer1_4 = torch.nn.Conv1d(in_channels=self.embedding_size,
                                             out_channels=64,
                                             kernel_size=5,
                                             stride=1
                                             )
        self.conv_layer2 = torch.nn.Conv1d(in_channels=128,
                                           out_channels=64,
                                           kernel_size=2,
                                           stride=1,
                                           dilation=2,
                                           padding=1
                                           )

        self.RNN_layer1 = nn.GRU(input_size=64,
                                 hidden_size=64,
                                 num_layers=1,
                                 bidirectional=True,
                                 batch_first=True
                                 )
        self.RNN_layer2_1 = nn.GRU(input_size=64,
                                   hidden_size=64,
                                   num_layers=1,
                                   bidirectional=True,
                                   batch_first=True
                                   )
        self.RNN_layer2_2 = nn.GRU(input_size=64,
                                   hidden_size=64,
                                   num_layers=1,
                                   bidirectional=True,
                                   batch_first=True
                                   )
        self.RNN_layer2_3 = nn.GRU(input_size=64,
                                   hidden_size=64,
                                   num_layers=1,
                                   bidirectional=True,
                                   batch_first=True
                                   )
        self.RNN_layer2_4 = nn.GRU(input_size=64,
                                   hidden_size=64,
                                   num_layers=1,
                                   bidirectional=True,
                                   batch_first=True
                                   )
        self.dropout = torch.nn.Dropout(0.6)

    def forward(self, x):
        # TCR1_1
        x1 = self.conv_layer1_1(x)
        x1 = torch.nn.ReLU()(x1)

        rnn_input1 = x1.permute(0, 2, 1)  # [batch_size, seq_len, feature]
        RNN_output1, _ = self.RNN_layer1(rnn_input1)
        c2_input = RNN_output1.permute(0, 2, 1)  # [batch_size, feature, seq_len]

        c2_out = self.conv_layer2(c2_input)
        r2_input = c2_out.permute(0, 2, 1)
        r2_out_1, _ = self.RNN_layer2_1(r2_input)
        out1 = RNN_output1 + r2_out_1
        out1 = out1.contiguous().view(out1.size(0), -1)

        # RC2
        x2 = self.conv_layer1_2(x)
        x2 = torch.nn.ReLU()(x2)

        rnn_input2 = x2.permute(0, 2, 1)
        RNN_output2, _ = self.RNN_layer1(rnn_input2)
        c2_input = RNN_output2.permute(0, 2, 1)  # [batch_size, feature, seq_len]

        c2_out = self.conv_layer2(c2_input)
        r2_input = c2_out.permute(0, 2, 1)
        r2_out_2, _ = self.RNN_layer2_2(r2_input)
        out2 = RNN_output2 + r2_out_2
        out2 = out2.contiguous().view(out2.size(0), -1)

        # RC3
        x3 = self.conv_layer1_3(x)
        x3 = torch.nn.ReLU()(x3)

        rnn_input3 = x3.permute(0, 2, 1)
        RNN_output3, _ = self.RNN_layer1(rnn_input3)
        c2_input = RNN_output3.permute(0, 2, 1)  # [batch_size, feature, seq_len]

        c2_out = self.conv_layer2(c2_input)
        r2_input = c2_out.permute(0, 2, 1)
        r2_out_3, _ = self.RNN_layer2_3(r2_input)
        out3 = RNN_output3 + r2_out_3
        out3 = out3.contiguous().view(out3.size(0), -1)

        out = torch.cat([out1, out2, out3], dim=-1)
        out = self.dropout(out)

        return out


class MLTP4(nn.Module):
    def __init__(self, vocab_size, embedding_size, output_size, dropout):
        super(MLTP4, self).__init__()

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

        self.maxpool1d_1 = torch.nn.MaxPool1d(kernel_size=1)
        self.maxpool1d_2 = torch.nn.MaxPool1d(kernel_size=1)
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
        self.attention = AdditiveAttention(input_size=128,
                                           value_size=21,
                                           num_hiddens=self.embedding_size,
                                           dropout=0.5)
        self.liner_out = nn.Sequential(
            # nn.Linear(18432, 2000), nn.ReLU(),
            # nn.Dropout(),
            nn.Linear(4032, 500), nn.ReLU(),
            nn.Dropout(),
            nn.Linear(500, 100), nn.ReLU(),
            nn.Dropout(),
            nn.Linear(100, self.output_size)
        )

        for i in range(21):
            setattr(self, "NaiveFC%d" % i, nn.Sequential(
                nn.Linear(in_features=192, out_features=48),
                nn.ReLU(),
                nn.Dropout(),
                nn.Linear(in_features=48, out_features=1),
                nn.Sigmoid()
            ))

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
        pool_out1 = self.maxpool1d_1(pool1_input)
        pool_out1 = pool_out1.permute(0, 2, 1)
        # rc1_out = pool_out.view(pool_out.size(0), -1)

        # RC2
        x2 = self.conv2(x)
        x2 = torch.nn.ReLU()(x2)
        rnn_input2 = x2.permute(0, 2, 1)
        RNN_output2, _ = self.BiRNN1(rnn_input2)
        pool2_input = RNN_output2.permute(0, 2, 1)  # [batch_size, feature, seq_len]
        pool_out2 = self.maxpool1d_2(pool2_input)
        pool_out2 = pool_out2.permute(0, 2, 1)
        # rc2_out = pool_out.view(pool_out.size(0), -1)

        # RC3
        x3 = self.conv3(x)
        x3 = torch.nn.ReLU()(x3)
        rnn_input3 = x3.permute(0, 2, 1)
        RNN_output3, _ = self.BiRNN1(rnn_input3)
        pool3_input = RNN_output3.permute(0, 2, 1)  # [batch_size, feature, seq_len]
        pool_out3 = self.maxpool1d_3(pool3_input)
        pool_out3 = pool_out3.permute(0, 2, 1)
        # rc3_out = pool_out.view(pool_out.size(0), -1)

        y = torch.cat([pool_out1, pool_out2, pool_out3], dim=1)  #
        y = self.dropout(y)

        return y

    def forward(self, input, valid_lens, in_feat=False):
        cnn_pssm2 = self.embed(input)
        # 嵌入层维度 torch.Size([batch, length, embedding_size])

        cnn_pssm2 = cnn_pssm2.permute(0, 2, 1)
        conv1d_input = torch.nn.ReLU()(cnn_pssm2)
        # 通过ReLU实现稀疏后的模型能够更好地挖掘相关特征，拟合训练数据
        # out = self.TR(con1d_input)
        # print(out.shape)
        cnn_vectors = self.cnn(conv1d_input)
        attention_out, attention_weight = self.attention(cnn_vectors, cnn_vectors, cnn_vectors)

        out_vectors = attention_out.view(attention_out.size(0), -1)
        # 线性label_out
        liner_label = self.liner_out(out_vectors)

        # 二分类label_out
        binary_outs = []
        for i in range(21):
            FClayer = getattr(self, "NaiveFC%d" % i)
            y = FClayer(attention_out[:, i, :])
            y = torch.squeeze(y, dim=-1)
            binary_outs.append(y)

        return liner_label, binary_outs
