#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/2/5 16:44
# @Author  : zdj
# @FileName: gettsv.py.py
# @Software: PyCharm
import pandas as pd
import os
train_src = ["hgmd", "gnomAD", "Pos_star2", "Neg_star2", "Pos_star1", "Neg_star1"]
pos_src = ["hgmd", "Pos_star2", "ddd_pos", "Pos_star1", "hgmd_test"]
neg_src = ["gnomAD", "Neg_star2", "ddd_neg", "Neg_star1", "gnomAD_test"]
clinvar_neg_src = [ "Neg_star2", "Neg_star1"]
test_src1 = ["ddd_neg", "ddd_pos"]
test_src2 = ["hgmd_test", "gnomAD_test"]
k_mer = 5
crop_len_list = [50, 70, 100, 150, 200, 225]

def write2tsv_single(kmer_seq, out_file, src):
    # print(kmer)
    assert (src in train_src) or (src in test_src1) or (src in test_src2)
    tag = 0
    if src in pos_src:
        tag = 1

    with open(out_file, 'a+') as f:
        f.write(kmer_seq + "\t" + str(tag) + "\n")

def data2tsv_fea(data, out_file, k_mer):
    # if not os.path.exists(out_file):
    #     os.makedirs(out_file)
    with open(out_file, 'w') as f:
        f.write("sequence	label\n")
    data.apply(lambda x: write2tsv_single(x[str(k_mer) + "mer_after"], out_file, x["src"]), axis=1)


def seq2kmer(seq, k):
    """
    Convert original sequence to kmers

    Arguments:
    seq -- str, original sequence.
    k -- int, kmer of length k specified.

    Returns:
    kmers -- str, kmers separated by space

    """
    #     print(len(seq))
    # assert len(seq)== seq_len
    kmer = [seq[x:x + k] for x in range(len(seq) + 1 - k)]
    kmers = " ".join(kmer)
    return kmers

def do_kmer(data, k):
    data[str(k) + "mer_after"] = data.apply(lambda x:seq2kmer(x["after_mutation"], k), axis = 1)
    return



def check_mutation_context(ref, alt, before, after, crop_len):
    if len(ref) != 1 and len(alt) != 1:
        global count_mul
        count_mul +=1
        return
    assert before[:crop_len] == after[:crop_len], 'before[:crop_len]: %s \n after[:crop_len] %s \n %s \n %s' % (before[:crop_len], after[:crop_len], before, after)
    assert before[-crop_len:] == after[-crop_len:], 'before[-crop_len:]: %s \n after[-crop_len:] %s \n %s \n %s ' % (before[-crop_len:], after[-crop_len:], before, after)
    if len(ref) == 1:
        assert after[crop_len: -crop_len] == alt[1:], ' %s \n %s '% (after[crop_len: -crop_len] , alt[1:])
    if len(alt) == 1:
        assert before[crop_len: -crop_len] == ref[1:], ' %s \n %s '% (before[crop_len: -crop_len] , ref[1:])

count_mul = 0

def check_mutation(pos, chr, ref, alt, before, after, crop_len):
    if len(ref) != 1 and len(alt) != 1:
        global count_mul
        count_mul +=1
        return
    print( pos)
    print(chr)
    print(before)
    print(after)
    assert before[:crop_len] == after[:crop_len], 'before[:crop_len]: %s \n after[:crop_len] %s \n %s \n %s' % (before[:crop_len], after[:crop_len], before, after)
    # assert before[-crop_len:] == after[-crop_len:], 'before[-crop_len:]: %s \n after[-crop_len:] %s \n %s \n %s ' % (before[-crop_len:], after[-crop_len:], before, after)
    mut_len = 0
    if len(ref) == 1:
        mut_len = len(alt[1:])
        assert after[crop_len: crop_len + mut_len] == alt[1:], ' %s \n %s '% (after[crop_len: crop_len + mut_len] , alt[1:])
        tmp_Len = len(after[crop_len + mut_len: ] )
        assert after[crop_len + mut_len: ] == before[crop_len : crop_len + tmp_Len]
    if len(alt) == 1:
        mut_len = len(ref[1:])
        assert before[crop_len: crop_len + mut_len] == ref[1:], ' %s \n %s '% (before[crop_len: crop_len + mut_len], ref[1:])
        tmp_Len = len(before[crop_len + mut_len: ] )
        assert before[crop_len + mut_len: ] == after[crop_len: crop_len + tmp_Len]


context = False
# vcf_file_path_list = ["DA_target.vcf", "DA_target_test.vcf"]
for k_mer in range(4,7):
    print(k_mer)
    for crop_len in crop_len_list:
        if context:
            datapath = os.path.join(str("context"), str(k_mer), str(crop_len))
            name = "_context.csv"
        else:
            datapath = os.path.join(str("NotContext"), str(k_mer), str(crop_len))
            name = ".csv"

        if not os.path.exists(datapath):
            os.makedirs(datapath)

        train_data = pd.read_csv("train" + str(crop_len) + name)
        # test_date = pd.read_csv("DA_target_test" + str(crop_len) + name)
        # DDD_data = pd.read_csv("DDD" + str(crop_len) + name)

        # print(train_data["before_mutation"].isnull().value_counts())
        # exit(1)
        train_data.apply(lambda x: check_mutation(x["POS"], x["CHROM"], x["REF"] , x["ALT"], x["before_mutation"],  x["after_mutation"], crop_len), axis = 1)

        # do_kmer(test_date, k_mer)
        do_kmer(train_data, k_mer)
        # do_kmer(DDD_data, k_mer)

        data2tsv_fea(train_data,  datapath + "/train.tsv", k_mer)
        # data2tsv_fea(test_date, datapath + "/dev.tsv", k_mer)
        # data2tsv_fea(DDD_data, datapath + "/DDD.tsv", k_mer)
