#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2023/2/27 15:00
# @Author : fhh
# @FileName: my_util.py
# @Software: PyCharm
import argparse
import numpy as np
import pandas as pd
from typing import Optional
import torch
import torch.nn as nn
import os

from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, TensorDataset, Dataset
import csv
import time
bases_kmer = 'ATCG'
def get_config():
    parse = argparse.ArgumentParser(description='default config')
    parse.add_argument('-do_train', action='store_true',
                       help=' ')
    parse.add_argument('-do_predict', action='store_true',
                       help=' ')
    # 数据参数
    parse.add_argument('-max_length', type=int, default=512,
                       help='Maximum length of peptide sequence')
    parse.add_argument('-vocab_size', type=int, default=4**6+1,
                       help='The size of the vocabulary')
    parse.add_argument('-output_size', type=int, default=1,
                       help='Number of peptide functions')

    parse.add_argument('-k_mer', type=int, default=0,
                       help='k of k-mer input data')

    parse.add_argument('-divide_validata', action='store_true',
                       help='divide 20% traindata to validata ')
    parse.add_argument('-dataset_direction', type=str, default=None,
                       help="dataset used to cross_validator")

    parse.add_argument('-load_npy', action='store_true')
    parse.add_argument('-confusion', action='store_true')

    # 训练参数
    parse.add_argument('-model_num', type=int, default=1,
                       help='Number of primary training models')
    parse.add_argument('-batch_size', type=int, default=64*4,
                       help='Batch size')
    parse.add_argument('-epochs', type=int, default=100)
    parse.add_argument('-learning_rate', type=float, default=0.0018)
    parse.add_argument('-threshold', type=float, default=0.5)
    parse.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parse.add_argument('--wd', '--weight-decay', default=1e-3, type=float,
                        metavar='W', help='weight decay (default: 1e-3)',
                        dest='weight_decay'    )
    parse.add_argument('-early_stop', type=int, default=5)
    parse.add_argument("-opt", type=str, default="SGD", help="optimizer")
    parse.add_argument('-k_fold', type=int, default=5)
    # 模型参数
    parse.add_argument('-attentionENcode', action='store_true',
                       help='use attention in ENcode')
    parse.add_argument('-model_name', type=str, default='TextCNN',
                       help='Name of the model')
    parse.add_argument('-embedding_size', type=int, default=64*4,
                       help='Dimension of the embedding')
    parse.add_argument('-dropout', type=float, default=0.6)
    parse.add_argument('-filter_num', type=int, default=64*2,
                       help='Number of the filter')
    parse.add_argument('-filter_size', nargs='+', type=int, default=[2, 3, 4, 5],
                       help='Size of the filter')

    # 路径参数
    parse.add_argument('-data_direction', type=str, default='/data3/linming/DNA_Lin/esm/scripts/data/',
                       help='Path of the training data')

    parse.add_argument('-dna_data_direction', type=str, default='/data3/linming/DNA_Lin/dataCenter/NotContext/',
                       help='Path of the training data')
    parse.add_argument('-train_direction', type=str, default='/data3/linming/DNA_Lin/esm/scripts/data/train.npy',
                       help='Path of the training data')
    parse.add_argument('-test_direction', type=str, default='/data3/linming/DNA_Lin/esm/scripts/data/DDD.npy',
                       help='Path of the test data')
    parse.add_argument('-model_path', type=str, default='/data3/linming/DNA_Lin/saved_models/textCNN_example_confusion1.pth',
                       help='Path of model for predicting')

    config = parse.parse_args()
    return config

class GaussianKernel(nn.Module):
    r"""Gaussian Kernel Matrix

    Gaussian Kernel k is defined by

    .. math::
        k(x_1, x_2) = \exp \left( - \dfrac{\| x_1 - x_2 \|^2}{2\sigma^2} \right)

    where :math:`x_1, x_2 \in R^d` are 1-d tensors.

    Gaussian Kernel Matrix K is defined on input group :math:`X=(x_1, x_2, ..., x_m),`

    .. math::
        K(X)_{i,j} = k(x_i, x_j)

    Also by default, during training this layer keeps running estimates of the
    mean of L2 distances, which are then used to set hyperparameter  :math:`\sigma`.
    Mathematically, the estimation is :math:`\sigma^2 = \dfrac{\alpha}{n^2}\sum_{i,j} \| x_i - x_j \|^2`.
    If :attr:`track_running_stats` is set to ``False``, this layer then does not
    keep running estimates, and use a fixed :math:`\sigma` instead.

    Args:
        sigma (float, optional): bandwidth :math:`\sigma`. Default: None
        track_running_stats (bool, optional): If ``True``, this module tracks the running mean of :math:`\sigma^2`.
          Otherwise, it won't track such statistics and always uses fix :math:`\sigma^2`. Default: ``True``
        alpha (float, optional): :math:`\alpha` which decides the magnitude of :math:`\sigma^2` when track_running_stats is set to ``True``

    Inputs:
        - X (tensor): input group :math:`X`

    Shape:
        - Inputs: :math:`(minibatch, F)` where F means the dimension of input features.
        - Outputs: :math:`(minibatch, minibatch)`
    """

    def __init__(self, sigma: Optional[float] = None, track_running_stats: Optional[bool] = True,
                 alpha: Optional[float] = 1.):
        super(GaussianKernel, self).__init__()
        assert track_running_stats or sigma is not None
        self.sigma_square = torch.tensor(sigma * sigma) if sigma is not None else None
        self.track_running_stats = track_running_stats
        self.alpha = alpha

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        l2_distance_square = ((X.unsqueeze(0) - X.unsqueeze(1)) ** 2).sum(2)

        if self.track_running_stats:
            self.sigma_square = self.alpha * torch.mean(l2_distance_square.detach())

        return torch.exp(-l2_distance_square / (2 * self.sigma_square))
def send_to_device(tensor, device):
    """
    Recursively sends the elements in a nested list/tuple/dictionary of tensors to a given device.

    Args:
        tensor (nested list/tuple/dictionary of :obj:`torch.Tensor`):
            The data to send to a given device.
        device (:obj:`torch.device`):
            The device to send the data to

    Returns:
        The same data structure as :obj:`tensor` with all tensors sent to the proper device.
    """
    if isinstance(tensor, (list, tuple)):
        return type(tensor)(send_to_device(t, device) for t in tensor)
    elif isinstance(tensor, dict):
        return type(tensor)({k: send_to_device(v, device) for k, v in tensor.items()})
    elif not hasattr(tensor, "to"):
        return tensor
    return tensor.to(device)

# (self.data_dna[idx]), (self.data_protein[idx]), (self.data_label[idx])
class ForeverDataIterator:
    r"""A data iterator that will never stop producing data"""

    def __init__(self, data_loader: DataLoader, device=None):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)
        self.device = device

    def __next__(self):
        try:
            data = next(self.iter)
            if self.device is not None:
                data = send_to_device(data, self.device)
        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)
            if self.device is not None:
                data = send_to_device(data, self.device)
        return data

    def __len__(self):
        return len(self.data_loader)

def PadEncode(data, max_len):
    amino_acids = 'XACDEFGHIKLMNPQRSTVWY'
    data_e, label_e, seq_length, temp = [], [], [], []
    sign, b = 0, 0
    for i in range(len(data)):
        length = len(data[i])
        elemt, st = [], data[i].strip()
        for j in st:
            if j not in amino_acids:
                sign = 1
                break
            index = amino_acids.index(j)
            elemt.append(index)
            sign = 0

        if length <= max_len and sign == 0:
            temp.append(elemt)
            seq_length.append(len(temp[b]))
            b += 1
            elemt += [0] * (max_len - length)
            data_e.append(elemt)
    return np.array(data_e), np.array(seq_length)

def code_kmer(token, k):
    result = 0
    for i in range(k):
        result *= len(bases_kmer)
        result += bases_kmer.index(token[i])
    return result + 1


def code_kmer_seq(kmer_seq, k):
    i = 0
    j = k
    coded = []
    while j <= len(kmer_seq):
        token = kmer_seq[i:j]
        coded.append(code_kmer(token, k))
        i += (k+1)
        j += (k+1)
    return np.array(coded)

def PadEncode_kmer(data1, max_len=450, kmer=5):
    kmer_length = (max_len-kmer+1)* (kmer +1) - 1
    # df1 = pd.DataFrame(data1)
    # df2 = pd.DataFrame(data2)


    # 序列编码
    data_e, label_e, seq_length, temp = [], [], [], []
    sign, b = 0, 0


    for i in range(len(data1)):
        if (len(data1[i]) != kmer_length) :
            print(len(data1[i]))
            print(kmer_length)
            continue

        element1 = code_kmer_seq(data1[i], kmer)
        # element2 = code_kmer_seq(data2[i], kmer)

        # element1 += [0] * (kmer_max - length1)  # �?补齐序列长度
        # element2 += [0] * (kmer_max - length2)

        data_e.append(element1)

    print(np.array(data_e).shape)
    return torch.LongTensor(np.array(data_e))


def collate_cont(batch):
    half_batch_size = int(len(batch)/2)
    dna1_list = []
    dna2_list = []
    protein1_list = []
    protein2_list = []
    label_list = []
    label1_list = []
    label2_list = []
    for i in range(half_batch_size):
        j = i + half_batch_size
        dna1, protein1, label1 = batch[i][0], batch[i][1], batch[i][2]
        dna2, protein2, label2 = batch[j][0], batch[j][1], batch[j][2]
        dna1_list.append(dna1)
        dna2_list.append(dna2)
        protein1_list.append(protein1)
        protein2_list.append(protein2)
        label1_list.append(int(label1))
        label2_list.append(int(label2))
        label = (int(label1) ^ int(label2))
        label_list.append(label)
    dna1_list = torch.from_numpy(np.asarray(dna1_list))
    dna2_list = torch.from_numpy(np.asarray(dna2_list))
    protein1_list = torch.from_numpy(np.asarray(protein1_list))
    protein2_list = torch.from_numpy(np.asarray(protein2_list))
    label1_list = torch.from_numpy(np.asarray(label1_list))
    label2_list = torch.from_numpy(np.asarray(label2_list))
    label_list = torch.from_numpy(np.asarray(label_list))
    return  dna1_list, dna2_list, protein1_list, protein2_list, label1_list, label2_list, label_list


class NpyDataset(Dataset):
    def __init__(self, npy_file, label_file):
        self.labels = np.load(label_file)
        if npy_file.find("dna") != -1:
            embedding_size = 768
            seq_len = 200
        else:
            embedding_size = 1280
            seq_len = 1024
        self.data = np.memmap(npy_file, dtype=np.float32, mode='r',
                              shape=(self.labels.shape[0], seq_len, embedding_size))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # return torch.from_numpy(self.data[idx]), torch.from_numpy(self.labels[idx])
        return (self.data[idx]), (self.labels[idx])


class Dataset_confusion(Dataset):
    def __init__(self, protein, dna_file, label_file):
        self.data_dna = dna_file
        self.data_label = label_file
        self.data_protein = protein

    def __len__(self):
        return len(self.data_dna)

    def __getitem__(self, idx):
        return (self.data_dna[idx]), (self.data_protein[idx]), (self.data_label[idx])

class NpyDataset_confusion(Dataset):
    def __init__(self, protein_file, dna_file, label_file):
        self.data_dna = np.lib.format.open_memmap(dna_file)
        self.data_label = np.load(label_file)
        self.data_protein = np.memmap(protein_file, dtype=np.float32, mode="r", shape=(self.data_label.shape[0], 1024, 1280))##

    def __len__(self):
        return len(self.data_dna)

    def __getitem__(self, idx):
        return (self.data_dna[idx]), (self.data_protein[idx]), (self.data_label[idx])

class NpyDataset_unlabel_confusion(Dataset):
    def __init__(self, protein_file, dna_file, label_file):
        self.data_dna = np.lib.format.open_memmap(dna_file)
        self.data_label = np.load(label_file)
        self.data_protein = np.memmap(protein_file, dtype=np.float32, mode="r", shape=(self.data_label.shape[0], 1024, 1280))

    def __len__(self):
        return len(self.data_dna)

    def __getitem__(self, idx):
        return (self.data_dna[idx]), (self.data_protein[idx])

class NpyDataset_confusion_cap(Dataset):
    def __init__(self, protein_file, dna_file, label_file, fea_path):
        self.data_dna = np.lib.format.open_memmap(dna_file)
        self.data_label = np.load(label_file)
        self.data_protein = np.memmap(protein_file, dtype=np.float32, mode="r", shape=(self.data_label.shape[0], 1024, 1280))
        self.cap_fea =  torch.tensor(pd.read_csv(fea_path).to_numpy()).to(torch.float32)

    def __len__(self):
        return len(self.data_dna)

    def __getitem__(self, idx):
        return (self.data_dna[idx]), (self.data_protein[idx]), (self.data_label[idx]), (self.cap_fea[idx])

class NpyDataset_confusion_div(Dataset):
    def __init__(self, data_protein, data_dna, data_label):
        self.data_dna = data_dna
        self.data_label = data_label
        self.data_protein = data_protein

    def __len__(self):
        return len(self.data_dna)

    def __getitem__(self, idx):
        return (self.data_dna[idx]), (self.data_protein[idx]), (self.data_label[idx])

class NpyDataset_confusion_div_cap(Dataset):
    def __init__(self, data_protein, data_dna, data_label, data_protein_len):
        self.data_dna = data_dna
        self.data_label = data_label
        self.data_protein = data_protein
        self.protein_len = data_protein_len

    def __len__(self):
        return len(self.data_dna)

    def __getitem__(self, idx):
        return (self.data_dna[idx]), (self.data_protein[idx]), (self.data_label[idx]), (self.protein_len[idx])


def data_load_npy(data_path, train_path, test_path, batch=32):
    train_label = os.path.join(data_path, "train_labels.npy")
    dataset_train = NpyDataset(train_path, train_label)
    dataset_train = DataLoader(dataset_train, batch_size=batch, shuffle=True)
    #
    test_label = os.path.join(data_path, "test_labels.npy")
    dataset_test = NpyDataset(test_path, test_label)
    dataset_test = DataLoader(dataset_test, batch_size=batch, shuffle=False)

    return dataset_train, dataset_test

def data_load_confusion(data_path, dna_data_direction, batch=32):
    train_protein_data_path = os.path.join(data_path, "train_protein.csv")
    train_dna_data_path = os.path.join(dna_data_direction, "train.tsv")
    train_protein_data = pd.read_csv(train_protein_data_path)
    train_label = train_protein_data["label"]
    protein_seq = train_protein_data["protein_seq"]
    protein_seq, _ = PadEncode(protein_seq, 1022)
    train_dna_data = pd.read_csv(train_dna_data_path, sep="\t")["sequence"]
    train_dna_data = PadEncode_kmer(train_dna_data)

    dataset_train = Dataset_confusion(protein_seq, train_dna_data, train_label)
    dataset_loader_train = DataLoader(dataset_train, batch_size=batch, shuffle=True)


    test_protein_data_path = os.path.join(data_path, "test_protein.csv")
    test_dna_data_path = os.path.join(dna_data_direction, "dev.tsv")
    test_protein_data = pd.read_csv(test_protein_data_path)
    test_label = test_protein_data["label"]
    protein_seq = test_protein_data["protein_seq"]
    protein_seq, _ = PadEncode(protein_seq, 1022)
    test_dna_data = pd.read_csv(test_dna_data_path, sep="\t")["sequence"]
    test_dna_data = PadEncode_kmer(test_dna_data)

    dataset_test = Dataset_confusion(protein_seq, test_dna_data, test_label)
    dataset_loader_test = DataLoader(dataset_test, batch_size=batch, shuffle=False)

    return dataset_loader_train, dataset_loader_test

def data_load_confusion_k_fold(data_path, dna_data_direction, batch=32):
    train_protein_data_path = os.path.join(data_path, "train_protein.csv")
    train_dna_data_path = os.path.join(dna_data_direction, "train.tsv")
    train_protein_data = pd.read_csv(train_protein_data_path)
    data_label = train_protein_data["label"]
    protein_seq = train_protein_data["protein_seq"]
    protein_seq, _ = PadEncode(protein_seq, 1022)
    train_dna_data1 = pd.read_csv(train_dna_data_path, sep="\t")["sequence"]
    data_dna = PadEncode_kmer(train_dna_data1)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    datasets_train, datasets_dev = [], []
    for i, (train_index, dev_index) in enumerate(cv.split(data_dna, data_label)):
        train_protein_data, train_dna_data, train_label = protein_seq[train_index], data_dna[train_index], data_label[
            train_index]
        dataset_train = NpyDataset_confusion_div(train_protein_data, train_dna_data, train_label)
        dev_protein_data, dev_dna_data, dev_label = protein_seq[dev_index], data_dna[dev_index], data_label[
            dev_index]

        dataset_dev = NpyDataset_confusion_div(dev_protein_data, dev_dna_data, dev_label)

        dataset_loader_train = DataLoader(dataset_train, batch_size=batch, shuffle=True)
        dataset_loader_dev = DataLoader(dataset_dev, batch_size=batch, shuffle=True)

        datasets_train.append(dataset_loader_train)
        datasets_dev.append(dataset_loader_dev)

    test_protein_data_path = os.path.join(data_path, "test_protein.csv")
    test_dna_data_path = os.path.join(dna_data_direction, "dev.tsv")
    test_protein_data = pd.read_csv(test_protein_data_path)
    test_label = test_protein_data["label"]
    protein_seq = test_protein_data["protein_seq"]
    protein_seq, _ = PadEncode(protein_seq, 1022)
    test_dna_data = pd.read_csv(test_dna_data_path, sep="\t")["sequence"]
    test_dna_data = PadEncode_kmer(test_dna_data)

    dataset_test = Dataset_confusion(protein_seq, test_dna_data, test_label)
    dataset_loader_test = DataLoader(dataset_test, batch_size=batch, shuffle=False)

    return datasets_train, datasets_dev, dataset_loader_test



def data_load_confusion_KD(data_path, dna_data_direction, batch=32):
    train_protein_data_path = os.path.join(data_path, "source_train_index.csv")
    train_dna_data_path = os.path.join(dna_data_direction, "source_train", "train.tsv")
    train_protein_data = pd.read_csv(train_protein_data_path)
    train_label = train_protein_data["label"]
    protein_seq = train_protein_data["protein_seq"]
    protein_seq, _ = PadEncode(protein_seq, 1022)
    train_dna_data = pd.read_csv(train_dna_data_path, sep="\t")["sequence"]
    train_dna_data = PadEncode_kmer(train_dna_data)

    dataset_train = Dataset_confusion(protein_seq, train_dna_data, train_label)
    dataset_loader_train_s = DataLoader(dataset_train, batch_size=batch, shuffle=True)


    test_protein_data_path = os.path.join(data_path, "test_protein.csv")
    test_dna_data_path = os.path.join(dna_data_direction, "dev.tsv")
    test_protein_data = pd.read_csv(test_protein_data_path)
    test_label = test_protein_data["label"]
    protein_seq = test_protein_data["protein_seq"]
    protein_seq, _ = PadEncode(protein_seq, 1022)
    test_dna_data = pd.read_csv(test_dna_data_path, sep="\t")["sequence"]
    test_dna_data = PadEncode_kmer(test_dna_data)

    dataset_test = Dataset_confusion(protein_seq, test_dna_data, test_label)
    dataset_loader_test_s = DataLoader(dataset_test, batch_size=batch, shuffle=False)


    train_label = os.path.join(data_path, "source_train_labels.npy")
    train_protein_data = os.path.join(data_path, "source_train.npy")
    train_dna_data = os.path.join(dna_data_direction, "source_train", "source_train_dna.npy")
    dataset_train = NpyDataset_confusion(train_protein_data, train_dna_data, train_label)
    dataset_loader_train_t = DataLoader(dataset_train, batch_size=batch, shuffle=True)

    # test_label = os.path.join(data_path, "test_labels.npy")
    # test_protein_data = os.path.join(data_path, "test.npy")
    # test_dna_data = os.path.join(dna_data_direction, "test_dna.npy")
    # dataset_test = NpyDataset_confusion(test_protein_data, test_dna_data, test_label)
    # dataset_test_t = DataLoader(dataset_test, batch_size=batch, shuffle=False)

    return dataset_loader_train_s, dataset_loader_test_s, dataset_loader_train_t

# def data_load_npy_confusion(data_path, batch=32):
#     train_label = os.path.join(data_path, "train_labels.npy")
#     train_protein_data = os.path.join(data_path, "train.npy")
#     train_dna_data = os.path.join(data_path, "train_dna.npy")
#     dataset_train = NpyDataset_confusion(train_protein_data, train_dna_data, train_label)
#     dataset_train = DataLoader(dataset_train, batch_size=batch, shuffle=True)
#     #
#     test_label = os.path.join(data_path, "test_labels.npy")
#     test_protein_data = os.path.join(data_path, "test.npy")
#     test_dna_data = os.path.join(data_path, "test_dna.npy")
#     dataset_test = NpyDataset_confusion(test_protein_data, test_dna_data, test_label)
#     dataset_test = DataLoader(dataset_test, batch_size=batch, shuffle=False)
#
#     return dataset_train, dataset_test

def data_load_npy_confusion_cont_cap(data_path, dna_data_direction, batch=32, is_cont=False):
    train_label = os.path.join(data_path, "train_labels.npy")
    train_protein_data = os.path.join(data_path, "train.npy")
    train_dna_data = os.path.join(dna_data_direction, "train_dna.npy")
    train_cap_fea = os.path.join(data_path, "cap_fea_train_norm.csv")
    dataset_train = NpyDataset_confusion_cap(train_protein_data, train_dna_data, train_label, train_cap_fea)
    dataset_loader_train = DataLoader(dataset_train, batch_size=batch, shuffle=True)
    #
    test_label = os.path.join(data_path, "test_labels.npy")
    test_protein_data = os.path.join(data_path, "test.npy")
    test_dna_data = os.path.join(dna_data_direction, "test_dna.npy")
    test_cap_fea = os.path.join(data_path, "cap_fea_test_norm.csv")
    dataset_test = NpyDataset_confusion_cap(test_protein_data, test_dna_data, test_label, test_cap_fea)
    dataset_test = DataLoader(dataset_test, batch_size=batch, shuffle=False)

    if is_cont:
        dataset_loader_train_cont = DataLoader(dataset_train, batch_size=batch, shuffle=True, collate_fn=collate_cont,
                                               drop_last=False)
    else:
        dataset_loader_train_cont = []
    return dataset_loader_train, dataset_loader_train_cont, dataset_test

def data_load_npy_confusion_cont(data_path, dna_data_direction, batch=32, is_cont = False):
    train_label = os.path.join(data_path, "train_labels.npy")
    train_protein_data = os.path.join(data_path, "train.npy")
    # train_dna_data = os.path.join(dna_data_direction, "train_dna.npy")
    # train_dna_data = os.path.join("/data2/liujie/linming/merged_features.npy")
    train_dna_data = os.path.join("/data2/liujie/linming/SMerged_features.npy")
    dataset_train = NpyDataset_confusion(train_protein_data, train_dna_data, train_label)
    dataset_loader_train = DataLoader(dataset_train, batch_size=batch, shuffle=True)

    if is_cont:
        dataset_loader_train_cont = DataLoader(dataset_train, batch_size=batch, shuffle=True, collate_fn=collate_cont,
                                               drop_last=False)
    else:
        dataset_loader_train_cont = []

    test_label = os.path.join(data_path, "test_labels.npy")
    test_protein_data = os.path.join(data_path, "test.npy")
    # test_dna_data = os.path.join(dna_data_direction, "test_dna.npy")
    # test_dna_data = os.path.join("/data2/liujie/linming/merged_features_test.npy")
    test_dna_data = os.path.join("/data2/liujie/linming/SMerged_features_test.npy")
    dataset_test = NpyDataset_confusion(test_protein_data, test_dna_data, test_label)
    dataset_test = DataLoader(dataset_test, batch_size=batch, shuffle=False)


    # test_label_path = os.path.join(data_path, "test_labels.npy")
    # test_protein_data_path = os.path.join(data_path, "test.npy")
    # test_dna_data_path = os.path.join(dna_data_direction, "test_dna.npy")
    # test_label = np.load(test_label_path)
    # test_protein_data = np.load(test_protein_data_path, allow_pickle=True)
    # test_dna_data = np.load(test_dna_data_path)
    #
    # dataset_test = NpyDataset_confusion_div(test_protein_data, test_dna_data, test_label)
    # dataset_test = DataLoader(dataset_test, batch_size=batch, shuffle=False)

    return dataset_loader_train, dataset_loader_train_cont, dataset_test


def data_load_npy_confusion_cont_source_train(data_path, dna_data_direction, batch=32, is_cont = False):
    train_label = os.path.join(data_path, "source_train_labels.npy")
    train_protein_data = os.path.join(data_path, "source_train.npy")
    train_dna_data = os.path.join(dna_data_direction, "source_train", "source_train_dna.npy")
    dataset_train = NpyDataset_confusion(train_protein_data, train_dna_data, train_label)
    dataset_loader_train = DataLoader(dataset_train, batch_size=batch, shuffle=True)

    if is_cont:
        dataset_loader_train_cont = DataLoader(dataset_train, batch_size=batch, shuffle=True, collate_fn=collate_cont,
                                               drop_last=False)
    else:
        dataset_loader_train_cont = []

    test_label = os.path.join(data_path, "test_labels.npy")
    test_protein_data = os.path.join(data_path, "test.npy")
    test_dna_data = os.path.join(dna_data_direction, "test_dna.npy")
    dataset_test = NpyDataset_confusion(test_protein_data, test_dna_data, test_label)
    dataset_test = DataLoader(dataset_test, batch_size=batch, shuffle=False)

    return dataset_loader_train, dataset_loader_train_cont, dataset_test


def data_load_npy_source(data_path, dna_data_direction, batch=32, is_cont = False):
    train_label = os.path.join(data_path, "source_Data_labels.npy")
    train_protein_data = os.path.join(data_path, "source_Data.npy")
    train_dna_data = os.path.join(dna_data_direction, "source", "source_dna.npy")
    dataset_train = NpyDataset_confusion(train_protein_data, train_dna_data, train_label)
    dataset_loader_train = DataLoader(dataset_train, batch_size=batch, shuffle=True, drop_last=True)

    return dataset_loader_train

#源域全部
def data_load_source(data_path, dna_data_direction, batch=32, is_cont = False):

    train_protein_data_path = os.path.join(data_path, "source_Data_index.csv")
    train_dna_data_path = os.path.join(dna_data_direction, "source", "train.tsv")
    train_protein_data = pd.read_csv(train_protein_data_path)
    train_label = train_protein_data["label"]
    protein_seq = train_protein_data["protein_seq"]
    protein_seq, _ = PadEncode(protein_seq, 1022)
    train_dna_data = pd.read_csv(train_dna_data_path, sep="\t")["sequence"]
    train_dna_data = PadEncode_kmer(train_dna_data)

    dataset_train = Dataset_confusion(protein_seq, train_dna_data, train_label)
    dataset_loader_train_s = DataLoader(dataset_train, batch_size=batch, shuffle=True, drop_last=True)

    return dataset_loader_train_s


# # target_labels.npy
# def data_load_npy_target(data_path, dna_data_direction, batch=32):
#     train_label = os.path.join(data_path, "target_labels.npy")
#     train_protein_data = os.path.join(data_path, "target.npy")
#     train_dna_data = os.path.join(dna_data_direction, "DA", "target_dna.npy")
#     dataset_train = NpyDataset_unlabel_confusion(train_protein_data, train_dna_data, train_label)
#     dataset_loader_train = DataLoader(dataset_train, batch_size=batch, shuffle=True, drop_last=True)

#     test_label = os.path.join(data_path, "DA_target_test_labels.npy")
#     test_protein_data = os.path.join(data_path, "DA_target_test.npy")
#     test_dna_data = os.path.join(dna_data_direction, "DA", "target_test_dna.npy")
#     dataset_test = NpyDataset_confusion(test_protein_data, test_dna_data, test_label)
#     dataset_test = DataLoader(dataset_test, batch_size=batch, shuffle=False)

#     return dataset_loader_train, dataset_test

def data_load_target(data_path, dna_data_direction, batch=32):

    train_protein_data_path = os.path.join(data_path, "target_index.csv")
    train_dna_data_path = os.path.join(dna_data_direction, "DA", "train.tsv")
    train_protein_data = pd.read_csv(train_protein_data_path)
    train_label = train_protein_data["label"]
    protein_seq = train_protein_data["protein_seq"]
    protein_seq, _ = PadEncode(protein_seq, 1022)
    train_dna_data = pd.read_csv(train_dna_data_path, sep="\t")["sequence"]
    train_dna_data = PadEncode_kmer(train_dna_data)

    dataset_train = Dataset_confusion(protein_seq, train_dna_data, train_label)
    train_target_dataset = DataLoader(dataset_train, batch_size=batch, shuffle=False, drop_last=True)


    test_protein_data_path = os.path.join(data_path, "target_test_index.csv")
    test_dna_data_path = os.path.join(dna_data_direction, "DA", "dev.tsv")
    test_protein_data = pd.read_csv(test_protein_data_path)
    test_label = test_protein_data["label"]
    protein_seq = test_protein_data["protein_seq"]
    protein_seq, _ = PadEncode(protein_seq, 1022)
    test_dna_data = pd.read_csv(test_dna_data_path, sep="\t")["sequence"]
    test_dna_data = PadEncode_kmer(test_dna_data)

    dataset_test = Dataset_confusion(protein_seq, test_dna_data, test_label)
    test_target_dataset = DataLoader(dataset_test, batch_size=batch, shuffle=False)

    return train_target_dataset, test_target_dataset

# todo: add strlen for cont
# 划分
def data_load_npy_confusion_cont_k_fold(data_path, dna_data_direction, k_fold, batch=32 , is_cont=False, n_splits = 5, DNAMode = "1"):
    cv = StratifiedKFold(n_splits = n_splits, shuffle=True, random_state=42)
    data_label_path = os.path.join(data_path, "train_labels.npy")
    data_protein_path = os.path.join(data_path, "train.npy")
    
    if DNAMode == "1":
        data_dna_path = os.path.join(dna_data_direction, "train_dna.npy")
        test_dna_data = os.path.join(dna_data_direction, "test_dna.npy")
    elif DNAMode == "2":
        data_dna_path = os.path.join("/data2/liujie/linming/merged_features.npy")
        test_dna_data = os.path.join("/data2/liujie/linming/merged_features_test.npy")
    elif DNAMode == "S":
        data_dna_path = os.path.join("/data2/liujie/linming/SMerged_features.npy")
        test_dna_data = os.path.join("/data2/liujie/linming/SMerged_features_test.npy")
    # save path
    protein_data_path = os.path.join(data_path, DNAMode, "train_cv.npy")
    dna_data_path = os.path.join(data_path, DNAMode, "train_dna_cv.npy")
    dev_protein_data_path = os.path.join(data_path, DNAMode, "dev_cv.npy")
    dev_dna_data_path = os.path.join(data_path, DNAMode, "dev_dna_cv.npy")   
    train_label_path = os.path.join(data_path,  DNAMode,"train_label_cv.npy")   
    dev_train_label_path = os.path.join(data_path, DNAMode, "dev_train_label_cv.npy")   

    data_label = np.load(data_label_path)
    data_dna = np.lib.format.open_memmap(data_dna_path)
    data_protein = np.memmap(data_protein_path, dtype=np.float32, mode="r",
                             shape=(data_label.shape[0], 1024, 1280))
    datasets_train, datasets_dev, datasets_cont = [], [], []
    for i, (train_index, dev_index) in enumerate(cv.split(data_dna, data_label)):
        if i != k_fold:
            continue
        train_protein_data, train_dna_data, train_label = data_protein[train_index], data_dna[train_index], data_label[
            train_index]
        dev_protein_data, dev_dna_data, dev_label = data_protein[dev_index], data_dna[dev_index], data_label[
            dev_index]
        np.save(train_label_path, train_label)
        np.save(dev_train_label_path, dev_label)
        np.save(protein_data_path, train_protein_data)
        np.save(dna_data_path, train_dna_data)
        np.save(dev_protein_data_path, dev_protein_data)
        np.save(dev_dna_data_path, dev_dna_data)
        return 
        dataset_train = NpyDataset_confusion_div(train_protein_data, train_dna_data, train_label)
        dataset_dev = NpyDataset_confusion_div(dev_protein_data, dev_dna_data, dev_label)

        dataset_loader_train = DataLoader(dataset_train, batch_size=batch, shuffle=True)
        if is_cont:
            dataset_loader_train_cont = DataLoader(dataset_train, batch_size=batch, shuffle=True, collate_fn=collate_cont,
                                                   drop_last=False)
        else:
            dataset_loader_train_cont = []
        dataset_loader_dev = DataLoader(dataset_dev, batch_size=batch, shuffle=True)

        datasets_train.append(dataset_loader_train)
        datasets_cont.append(dataset_loader_train_cont)
        datasets_dev.append(dataset_loader_dev)

    # test data
    test_label = os.path.join(data_path, "test_labels.npy")
    test_protein_data = os.path.join(data_path, "test.npy")
    # test_dna_data = os.path.join(dna_data_direction, "test_dna.npy")
    dataset_test = NpyDataset_confusion(test_protein_data, test_dna_data, test_label)
    dataset_test = DataLoader(dataset_test, batch_size=batch, shuffle=False)

    return datasets_train, datasets_cont, datasets_dev, dataset_test

# 读取
# TODO 不使用 open_memmap
def data_load_npy_confusion_cont_k_fold_divided(data_path, dna_data_direction, k_fold, batch=32 , is_cont=False, n_splits = 5, DNAMode = "S"):


    # save path
    protein_data_path = os.path.join(data_path, DNAMode, "train_cv.npy")
    dna_data_path = os.path.join(data_path, DNAMode, "train_dna_cv.npy")
    dev_protein_data_path = os.path.join(data_path, DNAMode, "dev_cv.npy")
    dev_dna_data_path = os.path.join(data_path, DNAMode, "dev_dna_cv.npy")   
    train_label_path = os.path.join(data_path,  DNAMode,"train_label_cv.npy")   
    dev_train_label_path = os.path.join(data_path, DNAMode, "dev_train_label_cv.npy")   

            
    datasets_train, datasets_dev, datasets_cont = [], [], []
    train_protein_data = np.lib.format.open_memmap(protein_data_path)
    train_dna_data = np.lib.format.open_memmap(dna_data_path)
    dev_protein_data = np.lib.format.open_memmap(dev_protein_data_path)
    dev_dna_data = np.lib.format.open_memmap(dev_dna_data_path)
    # train_protein_data = np.load(protein_data_path)
    # train_dna_data = np.load(dna_data_path)
    # dev_protein_data = np.load(dev_protein_data_path)
    # dev_dna_data = np.load(dev_dna_data_path)
    data_label = np.load(train_label_path)
    dev_label = np.load(dev_train_label_path)

    dataset_train = NpyDataset_confusion_div(train_protein_data, train_dna_data, data_label)
    dataset_dev = NpyDataset_confusion_div(dev_protein_data, dev_dna_data, dev_label)
    dataset_loader_train = DataLoader(dataset_train, batch_size=batch, shuffle=True)
    dataset_loader_dev = DataLoader(dataset_dev, batch_size=batch)

    datasets_train.append(dataset_loader_train)
    datasets_dev.append(dataset_loader_dev)

    # test data
    test_label = os.path.join(data_path, "test_labels.npy")
    test_protein_data = os.path.join(data_path, "test.npy")
    test_dna_data = os.path.join(dna_data_direction, "test_dna.npy")
    dataset_test = NpyDataset_confusion(test_protein_data, test_dna_data, test_label)
    dataset_test = DataLoader(dataset_test, batch_size=batch, shuffle=False)
    # dataset_test = []

    return datasets_train, [], datasets_dev, dataset_test


def data_load_npy_confusion_cont_k_fold_cap(data_path, dna_data_direction, k_fold, batch=32):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    data_label_path = os.path.join(data_path, "train_labels.npy")
    data_protein_path = os.path.join(data_path, "train.npy")
    data_dna_path = os.path.join(dna_data_direction, "train_dna.npy")
    data_protein_len_path = os.path.join(data_path, "train_list.csv")

    data_label = np.load(data_label_path)
    data_dna = np.lib.format.open_memmap(data_dna_path)
    data_protein = np.memmap(data_protein_path, dtype=np.float32, mode="r",
                             shape=(data_label.shape[0], 1024, 1280))
    data_protein_len = pd.read_csv(data_protein_len_path)["strlen"]
    datasets_train, datasets_dev, datasets_cont = [], [], []
    for i, (train_index, dev_index) in enumerate(cv.split(data_dna, data_label)):
        if i != k_fold:
            continue
        train_protein_data, train_dna_data, train_label, train_protein_len = data_protein[train_index], data_dna[train_index], data_label[
            train_index], data_protein_len[train_index]
        dataset_train = NpyDataset_confusion_div(train_protein_data, train_dna_data, train_label, train_protein_len)
        dev_protein_data, dev_dna_data, dev_label, dev_protein_len = data_protein[dev_index], data_dna[dev_index], data_label[
            dev_index], data_protein_len[dev_index]
        dataset_dev = NpyDataset_confusion_div(dev_protein_data, dev_dna_data, dev_label, dev_protein_len)

        dataset_loader_train = DataLoader(dataset_train, batch_size=batch, shuffle=True)
        dataset_loader_train_cont = DataLoader(dataset_train, batch_size=batch, shuffle=True, collate_fn=collate_cont,
                                               drop_last=False)
        dataset_loader_dev = DataLoader(dataset_dev, batch_size=batch, shuffle=True)

        datasets_train.append(dataset_loader_train)
        datasets_cont.append(dataset_loader_train_cont)
        datasets_dev.append(dataset_loader_dev)

    # test data
    test_label = os.path.join(data_path, "test_labels.npy")
    test_protein_data = os.path.join(data_path, "test.npy")
    test_dna_data = os.path.join(dna_data_direction, "test_dna.npy")
    dataset_test = NpyDataset_confusion(test_protein_data, test_dna_data, test_label)
    dataset_test = DataLoader(dataset_test, batch_size=batch, shuffle=False)

    return datasets_train, datasets_cont, datasets_dev, dataset_test



def data_load_npy_predict(test_path, test_label, batch=32):
    dataset_test = NpyDataset(test_path, test_label)
    dataset_test = DataLoader(dataset_test, batch_size=batch, shuffle=False)

    return dataset_test


def data_load_npy_confusion_predict(data_path, dna_data_direction, batch=32):
    # test_label = os.path.join(data_path, "DDD_labels.npy")
    # test_protein_data = os.path.join(data_path, "DDD.npy")
    # test_dna_data = os.path.join(dna_data_direction, "DDD", "DDD_dna.npy")

    test_label = os.path.join(data_path, "test_labels.npy")
    test_protein_data = os.path.join(data_path, "test.npy")
    test_dna_data = os.path.join(dna_data_direction, "test_dna.npy")

    # test_label = os.path.join(data_path, "target_test_labels.npy")
    # test_protein_data = os.path.join(data_path, "target_test.npy")
    # test_dna_data = os.path.join(dna_data_direction, "DA",  "target_test_dna.npy")

    dataset_test = NpyDataset_confusion(test_protein_data, test_dna_data, test_label)
    dataset_test = DataLoader(dataset_test, batch_size=batch, shuffle=False)

    return dataset_test

def data_load_npy_confusion_predict_DDD(data_path, dna_data_direction, batch=32):
    test_label = os.path.join(data_path, "DDD_labels.npy")
    test_protein_data = os.path.join(data_path, "DDD.npy")
    test_dna_data = os.path.join(dna_data_direction, "DDD", "DDD_dna.npy")

    # test_label = os.path.join(data_path, "test_labels.npy")
    # test_protein_data = os.path.join(data_path, "test.npy")
    # test_dna_data = os.path.join(dna_data_direction, "test_dna.npy")

    # test_label = os.path.join(data_path, "target_test_labels.npy")
    # test_protein_data = os.path.join(data_path, "target_test.npy")
    # test_dna_data = os.path.join(dna_data_direction, "DA",  "target_test_dna.npy")

    dataset_test = NpyDataset_confusion(test_protein_data, test_dna_data, test_label)
    dataset_test = DataLoader(dataset_test, batch_size=batch, shuffle=False)

    return dataset_test


def data_load_confusion_predict(data_path, dna_data_direction, batch=32):
    test_protein_data_path = os.path.join(data_path, "test_protein.csv")
    test_dna_data_path = os.path.join(dna_data_direction, "dev.tsv")
    test_protein_data = pd.read_csv(test_protein_data_path)
    test_label = test_protein_data["label"]
    protein_seq = test_protein_data["protein_seq"]
    protein_seq, _ = PadEncode(protein_seq, 1022)
    test_dna_data = pd.read_csv(test_dna_data_path, sep="\t")["sequence"]
    test_dna_data = PadEncode_kmer(test_dna_data)

    dataset_test = Dataset_confusion(protein_seq, test_dna_data, test_label)
    dataset_loader_test = DataLoader(dataset_test, batch_size=batch, shuffle=False)

    return dataset_loader_test

def data_load_confusion_predict_DDD(data_path, dna_data_direction, batch=32):
    test_protein_data_path = os.path.join(data_path, "DDD_protein.csv")
    test_dna_data_path = os.path.join(dna_data_direction, "DDD", "dev.tsv")
    test_protein_data = pd.read_csv(test_protein_data_path)
    test_label = test_protein_data["label"]
    protein_seq = test_protein_data["protein_seq"]
    protein_seq, _ = PadEncode(protein_seq, 1022)
    test_dna_data = pd.read_csv(test_dna_data_path, sep="\t")["sequence"]
    test_dna_data = PadEncode_kmer(test_dna_data)

    dataset_test = Dataset_confusion(protein_seq, test_dna_data, test_label)
    dataset_loader_test = DataLoader(dataset_test, batch_size=batch, shuffle=False)

    return dataset_loader_test

def mark_label(src):
    assert src.find("hgmd") >= 0 or src.find("pos") >= 0 or src.find("gnomAD") >= 0 or src.find("neg") >= 0 or src.find(
        "Neg") >= 0 or src.find("Pos") >= 0

    if src.find("hgmd") >= 0 or src.find("pos") >= 0 or src.find("Pos") >= 0:
        return 1
    return 0


def spent_time(start, end):
    epoch_time = end - start
    minute = int(epoch_time / 60)  # 分钟
    secs = int(epoch_time - minute * 60)  # �?
    return minute, secs


# '%.3f' % test_score[5],
def save_results(model_name, start, end, test_score, file_path):
    # 保存模型结果 csv文件
    title = ['Model']
    title.extend(test_score.keys())
    title.extend(['RunTime', 'Test_Time'])

    now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    content_row = [model_name]
    for key in test_score:
        content_row.append('%.3f' % test_score[key])
    content_row.extend([[end - start], now])

    content = [content_row]

    if os.path.exists(file_path):
        data = pd.read_csv(file_path, header=None)
        one_line = list(data.iloc[0])
        if one_line == title:
            with open(file_path, 'a+', newline='') as t:  # newline用来控制空的行数
                writer = csv.writer(t)  # 创建一个csv的写入器
                writer.writerows(content)  # 写入数据
        else:
            with open(file_path, 'a+', newline='') as t:
                writer = csv.writer(t)
                writer.writerow(title)  # 写入标题
                writer.writerows(content)
    else:
        with open(file_path, 'a+', newline='') as t:
            writer = csv.writer(t)
            writer.writerow(title)
            writer.writerows(content)


def print_score(test_score):
    if type(test_score) == dict:
        for key in test_score:
            print(key + f': {test_score[key]:.3f}')
    else:
        print(test_score)