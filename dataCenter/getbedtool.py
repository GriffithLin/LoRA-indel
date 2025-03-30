# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/2/5 13:52
# @Author  : zdj
# @FileName: getbedtool.py.py
# @Software: PyCharm

import pandas as pd
import numpy as np
import re

def check_indel_length(ref, alt, data):
    if abs(len(ref)-len(alt)) %3 != 0:
        print(data)

def ref_alt(ref, alt):
    return ref + "_" + alt

def vcf2bed(vcf_file_path, bed_len):
    vcf =pd.read_csv(vcf_file_path, sep = '\t')
    vcf["#CHROM"] = vcf["#CHROM"].apply(lambda x: "chr" + str(x))
    chr_id = vcf["#CHROM"]
    start = vcf["POS"].astype(int)
    start = start -1
    # vcf["name"] = vcf.apply(lambda x:ref_alt(x["REF"], x["ALT"]) , axis=1
    vcf["name"] = vcf["INFO"]
    bed = pd.DataFrame([chr_id, start.astype(int) - bed_len, start.astype(int) + bed_len + 1, vcf["name"]])
    bed = bed.transpose()
    return bed

def tmp_find(strlen, classStr):
    a = re.findall("\d+", strlen)
    return a[0] + "_" + str(int(classStr))

def dataList2vcf(datapath, vcf_file_path):
    data_list = pd.read_csv(datapath)
    data_list["#CHROM"] = data_list["name"].apply(lambda x: "chr" + x.split("_")[0])
    data_list["POS"] = data_list["name"].apply(lambda x: x.split("_")[1])
    data_list["REF"] = data_list["name"].apply(lambda x: x.split("_")[2])
    data_list["ALT"] = data_list["name"].apply(lambda x: x.split("_")[3])
    data_list["ID"] = '.'
    data_list["QUAL"] = '.'
    data_list["FILTER"] = '.'
    data_list["INFO"] = data_list.apply(lambda x: tmp_find(x["strlen"], x["class"]), axis = 1)
    data_list.to_csv(vcf_file_path, index=False, mode="w", na_rep=".", sep="\t",
                      columns=["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO"])
bed_len = 300
# input_file = "train_list.csv"
# vcf_file_path = "train.vcf"

# dataList2vcf(input_file, vcf_file_path)
# train_bed = vcf2bed(vcf_file_path, bed_len)
# train_bed.to_csv(vcf_file_path[:-4] + "_" + str(bed_len) + ".bed", encoding='utf-8', sep='\t', index=False, header=0)
#
# input_file = "test_list.csv"
# vcf_file_path = "test.vcf"
# dataList2vcf(input_file, vcf_file_path)
# train_bed = vcf2bed(vcf_file_path, bed_len)
# train_bed.to_csv(vcf_file_path[:-4] + "_" + str(bed_len) + ".bed", encoding='utf-8', sep='\t', index=False, header=0)


# input_file = "DDD_list.csv"
# vcf_file_path = "DDD.vcf"
# dataList2vcf(input_file, vcf_file_path)
# train_bed = vcf2bed(vcf_file_path, bed_len)
# train_bed.to_csv(vcf_file_path[:-4] + "_" + str(bed_len) + ".bed", encoding='utf-8', sep='\t', index=False, header=0)


vcf_file_path = "DA_target.vcf"
train_bed = vcf2bed(vcf_file_path, bed_len)
train_bed.to_csv(vcf_file_path[:-4] + "_" + str(bed_len) + ".bed", encoding='utf-8', sep='\t', index=False, header=0)

vcf_file_path = "DA_target_test.vcf"
train_bed = vcf2bed(vcf_file_path, bed_len)
train_bed.to_csv(vcf_file_path[:-4] + "_" + str(bed_len) + ".bed", encoding='utf-8', sep='\t', index=False, header=0)

vcf_file_path = "source_Data.vcf"
train_bed = vcf2bed(vcf_file_path, bed_len)
train_bed.to_csv(vcf_file_path[:-4] + "_" + str(bed_len) + ".bed", encoding='utf-8', sep='\t', index=False, header=0)

vcf_file_path = "source_train.vcf"
train_bed = vcf2bed(vcf_file_path, bed_len)
train_bed.to_csv(vcf_file_path[:-4] + "_" + str(bed_len) + ".bed", encoding='utf-8', sep='\t', index=False, header=0)