#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2023/2/27 15:00
# @Author : fhh
# @FileName: config.py
# @Software: PyCharm
import argparse


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
