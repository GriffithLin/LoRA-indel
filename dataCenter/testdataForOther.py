#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/2/11 15:10
# @Author  : zdj
# @FileName: testdataForOther.py.py
# @Software: PyCharm
import pandas as pd
test_data_divided = pd.read_csv("test.vcf", sep= '\t')
test_data_divided.to_csv( "testdata_input_cadd.vcf",index=False,mode="w",na_rep=".",sep="\t",columns=["#CHROM", "POS", "ID", "REF", "ALT"])

test_data_divided["#UID"] = test_data_divided.index
test_data_divided["#UID"]  = test_data_divided["#UID"] .apply(lambda x: "uid" + str(x))
test_data_divided["Strand"] = "+"
test_data_divided["CHROM"] = test_data_divided["#CHROM"].apply(lambda x: x[3:])
test_data_divided.to_csv( "CRAVAT.vcf",index=False,mode="w",na_rep=".",sep="\t",columns=["#UID", "CHROM", "POS", "Strand", "REF", "ALT"])
