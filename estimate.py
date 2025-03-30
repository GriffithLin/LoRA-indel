#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2022/3/21 10:28
# @Author  : ywh
# @File    : estimate.py
# @Software: PyCharm
# import
from sklearn.metrics import matthews_corrcoef, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.metrics import precision_recall_curve,auc, accuracy_score
import numpy as np
import torch

# def auc(score_label, Y_True, threshold=0.55):


# TP FP TN FN Total (%) Sens Spec LR+ LR− PPV NPV AUC MCC



def evaluate(y_pred, y_test, th=0.5):
    y_predlabel = [(0 if item < th else 1) for item in y_pred]
    y_test = np.array([(0 if item < 1 else 1) for item in y_test])
    y_predlabel = np.array(y_predlabel)
    tn, fp, fn, tp = confusion_matrix(y_test, y_predlabel).flatten()
    SP = tn * 1.0 / ((tn + fp) * 1.0)
    # SN = tp * 1.0 / ((tp + fn) * 1.0)
    # MCC = matthews_corrcoef(y_test, y_predlabel)
    Recall = recall_score(y_test, y_predlabel)
    Precision = precision_score(y_test, y_predlabel)
    F1 = f1_score(y_test, y_predlabel)
    Acc = accuracy_score(y_test, y_predlabel)
    AUC = roc_auc_score(y_test, y_pred)
    precision_aupr, recall_aupr, _ = precision_recall_curve(y_test, y_pred)
    # AUPR = auc(recall_aupr, precision_aupr)
    evl_result = {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "total": tn + fp + fn + tp}
    evl_result.update(
        {"recall": Recall, "SPE": SP, "PRE": Precision,  "F1": F1, "ACC": Acc , "AUC": AUC,})

    return evl_result
    # return Recall, SN, SP, MCC, Precision, F1, Acc, AUC, AUPR, tp, fn, tn, fp

def evaluate_tsf(eval_preds):
    # print("eval_preds:", eval_preds)

    y_pred, y_test = eval_preds
    # print(y_pred)
    # y_pred = torch.sigmoid(torch.tensor(y_pred))
    y_pred = torch.tensor(y_pred)
    th = 0.5
    y_predlabel = [(0 if item < th else 1) for item in y_pred]
    y_test = np.array([(0 if item < 1 else 1) for item in y_test])
    y_predlabel = np.array(y_predlabel)
    tn, fp, fn, tp = confusion_matrix(y_test, y_predlabel).flatten()
    SP = tn * 1.0 / ((tn + fp) * 1.0)
    # SN = tp * 1.0 / ((tp + fn) * 1.0)
    # MCC = matthews_corrcoef(y_test, y_predlabel)
    Recall = recall_score(y_test, y_predlabel)
    Precision = precision_score(y_test, y_predlabel)
    F1 = f1_score(y_test, y_predlabel)
    Acc = accuracy_score(y_test, y_predlabel)
    AUC = roc_auc_score(y_test, y_pred)
    precision_aupr, recall_aupr, _ = precision_recall_curve(y_test, y_pred)
    # AUPR = auc(recall_aupr, precision_aupr)
    evl_result = {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "total": tn + fp + fn + tp}
    evl_result.update(
        {"recall": Recall, "SPE": SP, "PRE": Precision,  "F1": F1, "ACC": Acc , "AUC": AUC,})

    # print("evl_result:", evl_result)
    return evl_result

#
# def evaluate(score_label, Y_True, threshold=0.5):
#
#     evl_result = dict()
#
#     # 将预测概率分数转为标签
#     y_hat = score_label
#     for i in range(len(score_label)):
#
#         if score_label[i] < threshold:  # threshold
#             y_hat[i] = 0
#         else:
#             y_hat[i] = 1
#
#
#     acc = accuracy_score(Y_True, y_hat)
#
#     tn, fp, fn, tp = confusion_matrix(Y_True, y_hat).ravel()
#     evl_result = {"tp":tp, "fp":fp, "tn":tn,"fn":fn , "total":tn + fp + fn + tp }
#
#     # print("Matthews相关系数: "+str(matthews_corrcoef(Y_True,y_hat)))
#     # print('sensitivity/recall:',tp/(tp+fn))
#     # print('specificity:',tn/(tn+fp))
#     # print("F1值: "+str(f1_score(Y_True,y_hat)))
#     # print('false positive rate:',fp/(tn+fp))
#     # print('false discovery rate:',fp/(tp+fp))
#     # print('TN:',tn,'FP:',fp,'FN:',fn,'TP:',tp)
#
#     Sens = tp / (tp + fn)
#     PPV = tp/(tp + fp)
#     NPV = tn/(tn + fn)
#     AUC = roc_auc_score(Y_True, y_hat)
#     mcc = matthews_corrcoef(Y_True, y_hat)
#     Precision = precision_score(Y_True, y_hat)
#
#     # Recall = tp / (tp + fn)
#     Recall = recall_score(Y_True, y_hat)
#     specificity = tn * 1.0 / ((tn + fp) * 1.0)
#     F1 = f1_score(Y_True, y_hat)
#     false_positive_rate = fp / (tn + fp)
#     false_discovery_rate= fp / (tp + fp)
#
#     # evl_result.update({"Sens":Sens, "Spec": specificity, "PPV":PPV, "NPV":NPV, "AUC":auc_precision_recall, "MCC":mcc, "acc":acc})
#     evl_result.update(
#         {"recall": Recall, "SPE": specificity, "PRE": Precision,  "F1": F1, "ACC": acc , "AUC": AUC,})
#
#     return evl_result

# Recall	SPE	PRE	F1	ACC	AUC