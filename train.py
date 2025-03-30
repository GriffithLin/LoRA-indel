#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2023/2/26 8:52
# @Author : fhh
# @FileName: train.py
# @Software: PyCharm
import time
import torch
import math
import numpy as np
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import accuracy_score, precision_recall_curve,auc, roc_auc_score
import os
import shutil
import hiddenlayer as hl
import torch.nn as nn

import estimate
from models.model import MultipleKernelMaximumMeanDiscrepancy, JointMultipleKernelMaximumMeanDiscrepancy, \
    DomainAdversarialLoss
from my_util import ForeverDataIterator
import torch.nn.functional as F
from meter import AverageMeter, ProgressMeter

# import tensorboard
class DataTrain_confusion:
    def __init__(self, model, optimizer, criterion, criterion_cont, scheduler=None, device="cuda"):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_scheduler = scheduler
        self.device = device
        self.criterion_cont = criterion_cont

    def train_step(self, train_iter, test_iter, modelname, epochs=None, model_num=0, early_stop = 10000, threshold = 0.5,printTrainAcc = False, printTestAcc = True, has_val = False):
        steps = 1
        best_loss = 100000.
        best_loss_acc = 0.
        best_auc = 0
        bestlos_epoch = 0
        PATH = os.getcwd()
        best_model = os.path.join(PATH, 'saved_models', modelname + 'best.pth')
        best_model_dir  = os.path.join(PATH, 'saved_models')
        # 检查路径是否存在
        if not os.path.exists(best_model_dir):
            # 如果路径不存在，则创建路径
            os.makedirs(best_model_dir)
            
        history1 = hl.History()
        # canvas1 = hl.Canvas()
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            total_loss = 0
            alpha = 0.4
            for train_dna, train_protein, train_label in train_iter:
                # print(train_data.shape)
                self.model.train()  # 进入训练模式
                # 使数据与模型在同一设备中
                train_dna, train_protein, train_label = train_dna.to(self.device), train_protein.to(self.device), train_label.to(
                    self.device)
                # 模型预测

                _, y_hat_train = self.model(train_dna, train_protein)

                # 计算损失
                # unsqueeze插入一个维度
                loss = self.criterion(y_hat_train, train_label.float().unsqueeze(1))

                # 清空之前的梯度
                self.optimizer.zero_grad()
                # 反向传播损失
                loss.backward()
                # 更新参数
                self.optimizer.step()

                if self.lr_scheduler:
                    if self.lr_scheduler.__module__ == lr_scheduler.__name__:
                        # Using PyTorch In-Built scheduler
                        self.lr_scheduler.step()
                    else:
                        # Using custom defined scheduler
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.lr_scheduler(steps)

                total_loss += loss.item()
                steps += 1

            end_time = time.time()
            epoch_time = end_time - start_time

            if printTrainAcc:
                model_predictions, true_labels = predict_confusion(self.model, train_iter, device=self.device)
                for i in range(len(model_predictions)):
                    if model_predictions[i] < threshold:  # threshold
                        model_predictions[i] = 0
                    else:
                        model_predictions[i] = 1
                y_hat = model_predictions
                acc1 = accuracy_score(true_labels, y_hat)
            else:
                acc1 = 0
            if printTestAcc:
                model_predictions, true_labels = predict_confusion_DA(self.model, test_iter, device=self.device)
                # for i in range(len(model_predictions)):
                #     if model_predictions[i] < threshold:  # threshold
                #         model_predictions[i] = 0
                #     else:
                #         model_predictions[i] = 1
                # y_hat = model_predictions
                test_auc = roc_auc_score(true_labels, model_predictions)
            else:
                test_auc = 0
            print(f'Model {model_num + 1}|Epoch:{epoch:003} | Time:{epoch_time:.2f}s')
            print(
                f'Train loss:{total_loss / len(train_iter)} ')
            # print(f'Train acc:{acc1}')
            print(f'test_auc:{test_auc}')
            # 计算每个epoch和step的模型输出特征
            history1.log((epoch, steps),
                         train_loss=total_loss / len(train_iter),  # 训练集损失
                         train_acc=acc1,  #
                         test_auc=test_auc)  # 验证集精度
            # 可视化网络训练的过程
            # with canvas1:
            #     canvas1.draw_plot(history1["train_loss"])
            #     canvas1.draw_plot(history1["train_acc"])
            #     canvas1.draw_plot(history1["test_auc"])

            if not has_val:
                train_loss = total_loss / len(train_iter)
                if train_loss < best_loss:
                    torch.save(self.model.state_dict(), best_model)
                    best_loss = train_loss
                    best_loss_acc = acc1
                    bestlos_epoch = epoch
                    
                if (train_loss > best_loss) and (epoch - bestlos_epoch >= early_stop):
                    break
            else:
                if test_auc > best_auc:
                    torch.save(self.model.state_dict(), best_model)
                    best_auc = test_auc
                    bestlos_epoch = epoch

                if (test_auc < best_auc) and (epoch - bestlos_epoch >= early_stop):
                    break


        self.model.load_state_dict(torch.load(best_model))
        os.remove(best_model)
        print("best_loss = " + str(best_loss))
        print("best_loss_acc = " + str(best_loss_acc))

        # canvas1.save('./save_img/train_test_' + str(model_num + 1) + modelname + '.pdf')

    def train_step_DA(self, train_src_iter: ForeverDataIterator, train_tar_iter, test_tar_iter, Basic_distance, modelname, epochs=None, model_num=0, early_stop = 10000, threshold = 0.5, printTestAcc = True):
        steps = 1
        best_loss = 100000.
        best_loss_acc = 0.
        bestlos_epoch = 0
        PATH = os.getcwd()
        best_model = os.path.join(PATH, 'saved_models', modelname + 'best.pth')

        history1 = hl.History()
        canvas1 = hl.Canvas()

        # batch_time = AverageMeter('Time', ':4.2f')
        # data_time = AverageMeter('Data', ':3.1f')
        # losses = AverageMeter('Loss', ':3.2f')
        # trans_losses = AverageMeter('Trans Loss', ':5.4f')
        # cls_accs = AverageMeter('Cls Acc', ':3.1f')

        for epoch in range(1, epochs + 1):
            # progress = ProgressMeter(
            #     [batch_time, data_time, losses, trans_losses, cls_accs],
            #     prefix="Epoch: [{}]".format(epoch))

            start_time = time.time()
            total_loss = 0
            total_cls_loss = 0
            total_transfer_loss = 0
            alpha = 0.4
            for train_t_dna, train_t_protein in train_tar_iter:
                self.model.train()  # 进入训练模式
                Basic_distance.train()
                x_s_dna, x_s_protein, label_s = next(train_src_iter)
                # 使数据与模型在同一设备中
                x_s_dna = x_s_dna.to(self.device)
                x_s_protein = x_s_protein.to(self.device)
                label_s = label_s.to(self.device)
                train_t_dna, train_t_protein = train_t_dna.to(self.device), train_t_protein.to(self.device)
                # 模型预测

                f_t, y_t = self.model(train_t_dna, train_t_protein)
                f_s, y_s = self.model(x_s_dna, x_s_protein)
                print(f_t.shape)
                print(y_t.shape)
                # 计算损失
                # unsqueeze插入一个维度
                cls_loss = self.criterion(y_s, label_s.float().unsqueeze(1))
                if type(Basic_distance) == MultipleKernelMaximumMeanDiscrepancy:
                    transfer_loss =Basic_distance(f_s, f_t)
                else:
                    transfer_loss = Basic_distance(
                        (f_s, F.softmax(y_s, dim=1)),
                        (f_t, F.softmax(y_t, dim=1))
                    )
                loss = cls_loss + transfer_loss * 1

                # 清空之前的梯度
                self.optimizer.zero_grad()
                # 反向传播损失
                loss.backward()
                # 更新参数
                self.optimizer.step()

                if self.lr_scheduler:
                    if self.lr_scheduler.__module__ == lr_scheduler.__name__:
                        # Using PyTorch In-Built scheduler
                        self.lr_scheduler.step()
                    else:
                        # Using custom defined scheduler
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.lr_scheduler(steps)

                total_loss += loss.item()
                total_cls_loss += cls_loss.item()
                total_transfer_loss += transfer_loss.item()
                steps += 1

            end_time = time.time()
            epoch_time = end_time - start_time

            if printTestAcc:
                model_predictions, true_labels = predict_confusion(self.model, test_tar_iter, device=self.device)
                for i in range(len(model_predictions)):
                    if model_predictions[i] < threshold:  # threshold
                        model_predictions[i] = 0
                    else:
                        model_predictions[i] = 1
                y_hat = model_predictions
                acc2 = accuracy_score(true_labels, y_hat)
            else:
                acc2 = 0
            print(f'Model {model_num + 1}|Epoch:{epoch:003} | Time:{epoch_time:.2f}s')
            print(
                f'Train loss:{total_loss / len(train_tar_iter)} ')
            print(
                f'total_cls_loss:{total_cls_loss / len(train_tar_iter)} ')
            print(
                f'total_transfer_loss:{total_transfer_loss / len(train_tar_iter)} ')
            # print(f'Train acc:{acc1}')
            print(f'Test acc:{acc2}')
            # 计算每个epoch和step的模型输出特征
            history1.log((epoch, steps),
                         train_loss=total_loss / len(train_iter),  # 训练集损失
                         test_acc=acc2)  # 验证集精度
            # 可视化网络训练的过程
            with canvas1:
                canvas1.draw_plot(history1["train_loss"])
                canvas1.draw_plot(history1["test_acc"])

            train_loss = total_loss / len(train_tar_iter)
            if train_loss < best_loss:
                torch.save(self.model.state_dict(), best_model)
                best_loss = train_loss
                bestlos_epoch = epoch


            if (best_loss < train_loss) and (epoch - bestlos_epoch >= early_stop):
                break



        self.model.load_state_dict(torch.load(best_model))
        os.remove(best_model)
        print("best_loss = " + str(best_loss))
        print("best_loss_acc = " + str(best_loss_acc))

        canvas1.save('./save_img/train_test_' + str(model_num + 1) + modelname + '.pdf')

    def train_step_DA_student(self, train_src_iter: ForeverDataIterator, train_tar_iter, test_tar_iter, Basic_distance, modelname, epochs=None, model_num=0, early_stop = 10000, threshold = 0.5, printTestAcc = True, trade_off = 1):
        steps = 1
        best_loss = 100000.
        best_loss_acc = 0.
        bestlos_epoch = 0
        PATH = os.getcwd()
        best_model = os.path.join(PATH, 'saved_models', modelname + 'best.pth')

        history1 = hl.History()
        canvas1 = hl.Canvas()

        # batch_time = AverageMeter('Time', ':4.2f')
        # data_time = AverageMeter('Data', ':3.1f')
        # losses = AverageMeter('Loss', ':3.2f')
        # trans_losses = AverageMeter('Trans Loss', ':5.4f')
        # cls_accs = AverageMeter('Cls Acc', ':3.1f')

        for epoch in range(1, epochs + 1):
            # progress = ProgressMeter(
            #     [batch_time, data_time, losses, trans_losses, cls_accs],
            #     prefix="Epoch: [{}]".format(epoch))

            start_time = time.time()
            total_loss = 0
            total_discri = 0
            total_cls_loss = 0
            total_transfer_loss = 0
            alpha = 0.4
            for train_t_dna, train_t_protein,_ in train_tar_iter:
                self.model.train()  # 进入训练模式
                Basic_distance.train()
                x_s_dna, x_s_protein, label_s = next(train_src_iter)
                # 使数据与模型在同一设备中
                x_s_dna = x_s_dna.to(self.device)
                x_s_protein = x_s_protein.to(self.device)
                label_s = label_s.to(self.device)
                train_t_dna, train_t_protein = train_t_dna.to(self.device), train_t_protein.to(self.device)
                # 模型预测

                f_t, y_t = self.model.cls(train_t_dna, train_t_protein)
                f_s, y_s = self.model.cls(x_s_dna, x_s_protein)
                # 计算损失
                # unsqueeze插入一个维度
                cls_loss = self.criterion(y_s, label_s.float().unsqueeze(1))
                if type(Basic_distance) == JointMultipleKernelMaximumMeanDiscrepancy:
                    transfer_loss = Basic_distance(
                        (f_s, F.softmax(y_s, dim=1)),
                        (f_t, F.softmax(y_t, dim=1))
                    )
                else:
                    transfer_loss = Basic_distance(f_s, f_t)
                loss = cls_loss + transfer_loss * trade_off
                if DomainAdversarialLoss == type(Basic_distance):
                    domain_acc = Basic_distance.domain_discriminator_accuracy

                # 清空之前的梯度
                self.optimizer.zero_grad()
                # 反向传播损失
                loss.backward()
                # 更新参数
                self.optimizer.step()

                if self.lr_scheduler:
                    if self.lr_scheduler.__module__ == lr_scheduler.__name__:
                        # Using PyTorch In-Built scheduler
                        self.lr_scheduler.step()
                    else:
                        # Using custom defined scheduler
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.lr_scheduler(steps)

                total_loss += loss.item()
                total_cls_loss += cls_loss.item()
                total_transfer_loss += transfer_loss.item()
                if DomainAdversarialLoss == type(Basic_distance):
                    total_discri += domain_acc
                steps += 1

            end_time = time.time()
            epoch_time = end_time - start_time

            if printTestAcc:
                model_predictions, true_labels = predict_confusion(self.model, test_tar_iter, device=self.device)
                for i in range(len(model_predictions)):
                    if model_predictions[i] < threshold:  # threshold
                        model_predictions[i] = 0
                    else:
                        model_predictions[i] = 1
                y_hat = model_predictions
                acc2 = accuracy_score(true_labels, y_hat)
            else:
                acc2 = 0
            print(f'Model {model_num + 1}|Epoch:{epoch:003} | Time:{epoch_time:.2f}s')
            print(
                f'Train loss:{total_loss / len(train_tar_iter)} ')
            print(
                f'total_cls_loss:{total_cls_loss / len(train_tar_iter)} ')
            print(
                f'total_transfer_loss:{total_transfer_loss / len(train_tar_iter)} ')
            if DomainAdversarialLoss == type(Basic_distance):
                print(
                    f'total_discri:{total_discri / len(train_tar_iter)} ')
            # print(f'Train acc:{acc1}')
            print(f'Test acc:{acc2}')
            # 计算每个epoch和step的模型输出特征
            history1.log((epoch, steps),
                         train_loss=total_loss / len(train_tar_iter),  # 训练集损失
                         test_acc=acc2, # 验证集精度
                         transfer_loss = total_transfer_loss / len(train_tar_iter))  
            # 可视化网络训练的过程
            with canvas1:
                canvas1.draw_plot(history1["train_loss"])
                canvas1.draw_plot(history1["test_acc"])
                canvas1.draw_plot(history1["transfer_loss"])

            train_loss = total_loss / len(train_tar_iter)
            if train_loss < best_loss:
                torch.save(self.model.state_dict(), best_model)
                best_loss = train_loss
                bestlos_epoch = epoch


            if (best_loss < train_loss) and (epoch - bestlos_epoch >= early_stop):
                break



        self.model.load_state_dict(torch.load(best_model))
        os.remove(best_model)
        print("best_loss = " + str(best_loss))
        print("best_loss_acc = " + str(best_loss_acc))

        canvas1.save('./save_img/train_test_' + str(model_num + 1) + modelname + '.pdf')

    def train_step_signle(self, train_iter, test_iter, modelname, epochs=None, model_num=0, early_stop = 10000, threshold = 0.5,printTrainAcc = False, printTestAcc = True):
        steps = 1
        best_loss = 100000.
        best_loss_acc = 0.
        bestlos_epoch = 0
        PATH = os.getcwd()
        best_model = os.path.join(PATH, 'saved_models', modelname + 'best.pth')

        history1 = hl.History()
        canvas1 = hl.Canvas()
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            total_loss = 0
            alpha = 0.4
            for train , train_label in train_iter:
                # print(train_data.shape)
                self.model.train()  # 进入训练模式
                # 使数据与模型在同一设备中
                train, train_label = train.to(self.device), train_label.to(
                    self.device)
                # 模型预测

                y_hat_train = self.model.cls(train)

                # 计算损失
                # unsqueeze插入一个维度
                loss = self.criterion(y_hat_train, train_label.float().unsqueeze(1))

                # 清空之前的梯度
                self.optimizer.zero_grad()
                # 反向传播损失
                loss.backward()
                # 更新参数
                self.optimizer.step()

                if self.lr_scheduler:
                    if self.lr_scheduler.__module__ == lr_scheduler.__name__:
                        # Using PyTorch In-Built scheduler
                        self.lr_scheduler.step()
                    else:
                        # Using custom defined scheduler
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.lr_scheduler(steps)

                total_loss += loss.item()
                steps += 1

            end_time = time.time()
            epoch_time = end_time - start_time

            if printTrainAcc:
                model_predictions, true_labels = predict(self.model, train_iter, device=self.device)
                for i in range(len(model_predictions)):
                    if model_predictions[i] < threshold:  # threshold
                        model_predictions[i] = 0
                    else:
                        model_predictions[i] = 1
                y_hat = model_predictions
                acc1 = accuracy_score(true_labels, y_hat)
            else:
                acc1 = 0
            if printTestAcc:
                model_predictions, true_labels = predict(self.model, test_iter, device=self.device)
                for i in range(len(model_predictions)):
                    if model_predictions[i] < threshold:  # threshold
                        model_predictions[i] = 0
                    else:
                        model_predictions[i] = 1
                y_hat = model_predictions
                acc2 = accuracy_score(true_labels, y_hat)
            else:
                acc2 = 0
            print(f'Model {model_num + 1}|Epoch:{epoch:003} | Time:{epoch_time:.2f}s')
            print(
                f'Train loss:{total_loss / len(train_iter)} ')
            # print(f'Train acc:{acc1}')
            print(f'Test acc:{acc2}')
            # 计算每个epoch和step的模型输出特征
            history1.log((epoch, steps),
                         train_loss=total_loss / len(train_iter),  # 训练集损失
                         train_acc=acc1,  #
                         test_acc=acc2)  # 验证集精度
            # 可视化网络训练的过程
            with canvas1:
                canvas1.draw_plot(history1["train_loss"])
                canvas1.draw_plot(history1["train_acc"])
                canvas1.draw_plot(history1["test_acc"])

            train_loss = total_loss / len(train_iter)
            if train_loss < best_loss:
                torch.save(self.model.state_dict(), best_model)
                best_loss = train_loss
                best_loss_acc = acc1
                bestlos_epoch = epoch


            if (best_loss < train_loss) and (epoch - bestlos_epoch >= early_stop):
                break



        self.model.load_state_dict(torch.load(best_model))
        os.remove(best_model)
        print("best_loss = " + str(best_loss))
        print("best_loss_acc = " + str(best_loss_acc))

        canvas1.save('./save_img/train_test_' + str(model_num + 1) + modelname + '.pdf')


    def train_step_batch_pad(self, train_iter, test_iter, modelname, epochs=None, model_num=0, early_stop = 10000, threshold = 0.5):
        steps = 1
        best_loss = 100000.
        best_loss_acc = 0.
        bestlos_epoch = 0
        PATH = os.getcwd()
        best_model = os.path.join(PATH, 'saved_models', modelname + 'best.pth')

        history1 = hl.History()
        canvas1 = hl.Canvas()
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            total_loss = 0
            alpha = 0.4
            for train_dna, train_protein, train_label, protein_len in train_iter:
                # print(train_data.shape)
                self.model.train()  # 进入训练模式
                # 使数据与模型在同一设备中
                train_dna, train_protein, train_label, protein_len = train_dna.to(self.device), train_protein.to(self.device), train_label.to(
                    self.device), protein_len.to(self.device)
                # 模型预测

                y_hat_train = self.model.cls(train_dna, train_protein, protein_len)

                # 计算损失
                # unsqueeze插入一个维度
                loss = self.criterion(y_hat_train, train_label.float().unsqueeze(1))

                # 清空之前的梯度
                self.optimizer.zero_grad()
                # 反向传播损失
                loss.backward()
                # 更新参数
                self.optimizer.step()

                if self.lr_scheduler:
                    if self.lr_scheduler.__module__ == lr_scheduler.__name__:
                        # Using PyTorch In-Built scheduler
                        self.lr_scheduler.step()
                    else:
                        # Using custom defined scheduler
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.lr_scheduler(steps)

                total_loss += loss.item()
                steps += 1

            end_time = time.time()
            epoch_time = end_time - start_time

            model_predictions, true_labels = predict_confusion_pad(self.model, train_iter, device=self.device)
            for i in range(len(model_predictions)):
                if model_predictions[i] < threshold:  # threshold
                    model_predictions[i] = 0
                else:
                    model_predictions[i] = 1
            y_hat = model_predictions
            acc1 = accuracy_score(true_labels, y_hat)

            model_predictions, true_labels = predict_confusion_pad(self.model, test_iter, device=self.device)
            for i in range(len(model_predictions)):
                if model_predictions[i] < threshold:  # threshold
                    model_predictions[i] = 0
                else:
                    model_predictions[i] = 1
            y_hat = model_predictions
            acc2 = accuracy_score(true_labels, y_hat)

            print(f'Model {model_num + 1}|Epoch:{epoch:003} | Time:{epoch_time:.2f}s')
            print(
                f'Train loss:{total_loss / len(train_iter)} ')
            print(f'Train acc:{acc1}')
            print(f'Test acc:{acc2}')
            # 计算每个epoch和step的模型输出特征
            history1.log((epoch, steps),
                         train_loss=total_loss / len(train_iter),  # 训练集损失
                         train_acc=acc1,  #
                         test_acc=acc2)  # 验证集精度
            # 可视化网络训练的过程
            with canvas1:
                canvas1.draw_plot(history1["train_loss"])
                canvas1.draw_plot(history1["train_acc"])
                canvas1.draw_plot(history1["test_acc"])

            train_loss = total_loss / len(train_iter)
            if train_loss < best_loss:
                torch.save(self.model.state_dict(), best_model)
                best_loss = train_loss
                best_loss_acc = acc1
                bestlos_epoch = epoch


            if (best_loss < train_loss) and (epoch - bestlos_epoch >= early_stop):
                break



        self.model.load_state_dict(torch.load(best_model))
        print("best_loss = " + str(best_loss))
        print("best_loss_acc = " + str(best_loss_acc))

        canvas1.save('./save_img/train_test_' + str(model_num + 1) + modelname + '.pdf')


    def train_step_cont(self, train_iter, train_cont_iter, test_iter, modelname, epochs=None, model_num=0, early_stop = 10000, threshold = 0.5):
        steps = 1
        best_loss = 100000.
        best_loss_acc = 0.
        bestlos_epoch = 0
        PATH = os.getcwd()
        best_model = os.path.join(PATH, 'saved_models', modelname + 'best.pth')
        canvas1 = hl.Canvas()
        history1 = hl.History()

        for epoch in range(1, epochs + 1):
            start_time = time.time()
            total_loss = 0
            total_loss_cont = 0
            total_loss1 = 0
            total_loss2 = 0
            alpha = 0.4
            for train_dna1, train_dna2, train_protein1, train_protein2, train_label1, train_label2, train_label_cont in train_cont_iter:
                # print(train_data.shape)
                self.model.train()  # 进入训练模式
                # 使数据与模型在同一设备中
                train_dna1, train_protein1, train_label1 = train_dna1.to(self.device), train_protein1.to(self.device), train_label1.to(
                    self.device)
                train_dna2, train_protein2, train_label2 = train_dna2.to(self.device), train_protein2.to(
                    self.device), train_label2.to(self.device)
                train_label_cont = train_label_cont.to(self.device)
                # 模型预测

                y_hat1 = self.model.cls(train_dna1, train_protein1)
                y_hat2 = self.model.cls(train_dna2, train_protein2)
                feature1 = self.model(train_dna1, train_protein1)
                feature2 = self.model(train_dna2, train_protein2)

                # 计算损失
                # unsqueeze插入一个维度
                loss1 = self.criterion(y_hat1, train_label1.float().unsqueeze(1))
                loss2 = self.criterion(y_hat2, train_label2.float().unsqueeze(1))
                loss_cont = self.criterion_cont(feature1, feature2, train_label_cont)
                loss = loss1 + loss2 + loss_cont
                # 清空之前的梯度
                self.optimizer.zero_grad()
                # 反向传播损失
                loss.backward()
                # 更新参数
                self.optimizer.step()

                if self.lr_scheduler:
                    if self.lr_scheduler.__module__ == lr_scheduler.__name__:
                        # Using PyTorch In-Built scheduler
                        self.lr_scheduler.step()
                    else:
                        # Using custom defined scheduler
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.lr_scheduler(steps)

                total_loss += loss.item()
                total_loss1 += loss1.item()
                total_loss2 += loss2.item()
                total_loss_cont += loss_cont.item()
                steps += 1

            end_time = time.time()
            epoch_time = end_time - start_time

            model_predictions, true_labels = predict_confusion(self.model, train_iter, device=self.device)
            for i in range(len(model_predictions)):
                if model_predictions[i] < threshold:  # threshold
                    model_predictions[i] = 0
                else:
                    model_predictions[i] = 1
            y_hat = model_predictions
            acc1 = accuracy_score(true_labels, y_hat)

            model_predictions, true_labels = predict_confusion(self.model, test_iter, device=self.device)
            for i in range(len(model_predictions)):
                if model_predictions[i] < threshold:  # threshold
                    model_predictions[i] = 0
                else:
                    model_predictions[i] = 1
            y_hat = model_predictions
            acc2 = accuracy_score(true_labels, y_hat)

            print(f'Model {model_num+1}|Epoch:{epoch:003} | Time:{epoch_time:.2f}s')
            print(f'Train loss:{total_loss / len(train_iter)}  loss1:{total_loss1 / len(train_iter)} loss2:{total_loss2 / len(train_iter)}  loss_cont:{total_loss_cont / len(train_iter)}')
            print(f'Train acc:{acc1}')
            print(f'Test acc:{acc2}')
            # 计算每个epoch和step的模型输出特征
            history1.log((epoch, steps),
                         train_loss=total_loss / len(train_iter),  # 训练集损失
                         train_acc=acc1, #
                         test_acc=acc2)  # 验证集精度
            # 可视化网络训练的过程
            with canvas1:
                canvas1.draw_plot(history1["train_loss"])
                canvas1.draw_plot(history1["train_acc"])
                canvas1.draw_plot(history1["test_acc"])

            train_loss = total_loss / len(train_iter)
            if train_loss < best_loss:
                torch.save(self.model.state_dict(), best_model)
                best_loss = train_loss
                best_loss_acc = acc1
                bestlos_epoch = epoch


            if (best_loss < train_loss) and (epoch - bestlos_epoch >= early_stop):
                break



        self.model.load_state_dict(torch.load(best_model))
        print("best_loss = " + str(best_loss))
        print("best_loss_acc = " + str(best_loss_acc))

        canvas1.save('./save_img/train_test_' + str(model_num + 1) + modelname + '.pdf')

    def train_step_cap(self, train_iter, test_iter, modelname, epochs=None, model_num=0, early_stop = 10000, threshold = 0.5):
        steps = 1
        best_loss = 100000.
        best_loss_acc = 0.
        bestlos_epoch = 0
        PATH = os.getcwd()
        best_model = os.path.join(PATH, 'saved_models', modelname + 'best.pth')

        history1 = hl.History()
        canvas1 = hl.Canvas()
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            total_loss = 0
            alpha = 0.4
            for train_dna, train_protein, train_label, capa_fea in train_iter:
                # print(train_data.shape)
                self.model.train()  # 进入训练模式
                # 使数据与模型在同一设备中
                train_dna, train_protein, train_label = train_dna.to(self.device), train_protein.to(self.device), train_label.to(
                    self.device)
                capa_fea = capa_fea.to(self.device)
                # 模型预测

                y_hat_train = self.model.cls(train_dna, train_protein, capa_fea)

                # 计算损失
                # unsqueeze插入一个维度
                loss = self.criterion(y_hat_train, train_label.float().unsqueeze(1))

                # 清空之前的梯度
                self.optimizer.zero_grad()
                # 反向传播损失
                loss.backward()
                # 更新参数
                self.optimizer.step()

                if self.lr_scheduler:
                    if self.lr_scheduler.__module__ == lr_scheduler.__name__:
                        # Using PyTorch In-Built scheduler
                        self.lr_scheduler.step()
                    else:
                        # Using custom defined scheduler
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.lr_scheduler(steps)

                total_loss += loss.item()
                steps += 1

            end_time = time.time()
            epoch_time = end_time - start_time

            # model_predictions, true_labels = predict_confusion_cap(self.model, train_iter, device=self.device)
            # for i in range(len(model_predictions)):
            #     if model_predictions[i] < threshold:  # threshold
            #         model_predictions[i] = 0
            #     else:
            #         model_predictions[i] = 1
            # y_hat = model_predictions
            # acc1 = accuracy_score(true_labels, y_hat)

            model_predictions, true_labels = predict_confusion_cap(self.model, test_iter, device=self.device)
            for i in range(len(model_predictions)):
                if model_predictions[i] < threshold:  # threshold
                    model_predictions[i] = 0
                else:
                    model_predictions[i] = 1
            y_hat = model_predictions
            acc2 = accuracy_score(true_labels, y_hat)

            print(f'Model {model_num + 1}|Epoch:{epoch:003} | Time:{epoch_time:.2f}s')
            print(
                f'Train loss:{total_loss / len(train_iter)} ')
            # print(f'Train acc:{acc1}')
            print(f'Test acc:{acc2}')
            # 计算每个epoch和step的模型输出特征
            history1.log((epoch, steps),
                         train_loss=total_loss / len(train_iter),  # 训练集损失
                         # train_acc=acc1,  #
                         test_acc=acc2)  # 验证集精度
            # 可视化网络训练的过程
            with canvas1:
                canvas1.draw_plot(history1["train_loss"])
                # canvas1.draw_plot(history1["train_acc"])
                canvas1.draw_plot(history1["test_acc"])

            train_loss = total_loss / len(train_iter)
            if train_loss < best_loss:
                torch.save(self.model.state_dict(), best_model)
                best_loss = train_loss
                # best_loss_acc = acc1
                bestlos_epoch = epoch


            if (best_loss < train_loss) and (epoch - bestlos_epoch >= early_stop):
                break



        self.model.load_state_dict(torch.load(best_model))
        print("best_loss = " + str(best_loss))
        print("best_loss_acc = " + str(best_loss_acc))

        canvas1.save('./save_img/train_test_' + str(model_num + 1) + modelname + '.pdf')


    def train_step_val(self, train_iter, val_iter, test_iter, modelname, epochs=None, model_num=0, early_stop=10000,
                   threshold=0.5):
        steps = 1
        best_loss = 100000.
        best_loss_acc = 0.
        bestlos_epoch = 0
        PATH = os.getcwd()
        best_model = os.path.join(PATH, 'saved_models', modelname + 'best.pth')

        history1 = hl.History()
        canvas1 = hl.Canvas()
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            total_loss = 0
            alpha = 0.4
            for train_dna, train_protein, train_label in train_iter:
                # print(train_data.shape)
                self.model.train()  # 进入训练模式
                # 使数据与模型在同一设备中
                train_dna, train_protein, train_label = train_dna.to(self.device), train_protein.to(
                    self.device), train_label.to(
                    self.device)
                # 模型预测

                y_hat = self.model.cls(train_dna, train_protein)

                # 计算损失
                # unsqueeze插入一个维度
                loss = self.criterion(y_hat, train_label.float().unsqueeze(1))

                # 清空之前的梯度
                self.optimizer.zero_grad()
                # 反向传播损失
                loss.backward()
                # 更新参数
                self.optimizer.step()

                if self.lr_scheduler:
                    if self.lr_scheduler.__module__ == lr_scheduler.__name__:
                        # Using PyTorch In-Built scheduler
                        self.lr_scheduler.step()
                    else:
                        # Using custom defined scheduler
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.lr_scheduler(steps)

                total_loss += loss.item()
                steps += 1

            end_time = time.time()
            epoch_time = end_time - start_time

            model_predictions, true_labels = predict_confusion(self.model, train_iter, device=self.device)
            for i in range(len(model_predictions)):
                if model_predictions[i] < threshold:  # threshold
                    model_predictions[i] = 0
                else:
                    model_predictions[i] = 1
            y_hat = model_predictions
            acc1 = accuracy_score(true_labels, y_hat)

            model_predictions, true_labels = predict_confusion(self.model, val_iter, device=self.device)
            for i in range(len(model_predictions)):
                if model_predictions[i] < threshold:  # threshold
                    model_predictions[i] = 0
                else:
                    model_predictions[i] = 1
            y_hat = model_predictions
            acc2 = accuracy_score(true_labels, y_hat)

            model_predictions, true_labels = predict_confusion(self.model, test_iter, device=self.device)
            for i in range(len(model_predictions)):
                if model_predictions[i] < threshold:  # threshold
                    model_predictions[i] = 0
                else:
                    model_predictions[i] = 1
            y_hat = model_predictions
            acc3 = accuracy_score(true_labels, y_hat)

            print(f'Model {model_num + 1}|Epoch:{epoch:003} | Time:{epoch_time:.2f}s')
            print(
                f'Train loss:{total_loss / len(train_iter)} ')
            print(f'Train acc:{acc1}')
            print(f'val acc:{acc2}')
            print(f'Test acc:{acc3}')
            # 计算每个epoch和step的模型输出特征
            history1.log((epoch, steps),
                         train_loss=total_loss / len(train_iter),  # 训练集损失
                         train_acc=acc1,
                         val_acc=acc2, #
                         test_acc=acc3)  # 验证集精度
            # 可视化网络训练的过程
            with canvas1:
                canvas1.draw_plot(history1["train_loss"])
                canvas1.draw_plot(history1["train_acc"])
                canvas1.draw_plot(history1["val_acc"])
                canvas1.draw_plot(history1["test_acc"])

            train_loss = total_loss / len(train_iter)
            if train_loss < best_loss:
                torch.save(self.model.state_dict(), best_model)
                best_loss = train_loss
                best_loss_acc = acc1
                bestlos_epoch = epoch

            if (best_loss < train_loss) and (epoch - bestlos_epoch >= early_stop):
                break

        self.model.load_state_dict(torch.load(best_model))
        print("best_loss = " + str(best_loss))
        print("best_loss_acc = " + str(best_loss_acc))

        canvas1.save('./save_img/train_test_' + str(model_num + 1) + modelname + '.pdf')


class DataTrain_confusion_DA:
    def __init__(self, model, model_target,  optimizer, criterion, scheduler=None, device="cuda"):
        self.model = model.to(device)
        self.model_target = model_target.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_scheduler = scheduler
        self.device = device


    def train_step_ADDA(self, train_iter, test_iter, modelname, epochs=None, model_num=0, early_stop = 10000, threshold = 0.5,printTrainAcc = False, printTestAcc = True):
        steps = 1
        best_loss = 100000.
        best_loss_acc = 0.
        bestlos_epoch = 0
        PATH = os.getcwd()
        best_model = os.path.join(PATH, 'saved_models', modelname + 'best.pth')

        history1 = hl.History()
        canvas1 = hl.Canvas()
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            total_loss = 0
            alpha = 0.4
            for train_dna, train_protein, train_label in train_iter:
                # print(train_data.shape)
                self.model.train()  # 进入训练模式
                # 使数据与模型在同一设备中
                train_dna, train_protein, train_label = train_dna.to(self.device), train_protein.to(self.device), train_label.to(
                    self.device)
                # 模型预测

                y_hat_train = self.model.cls(train_dna, train_protein)

                # 计算损失
                # unsqueeze插入一个维度
                loss = self.criterion(y_hat_train, train_label.float().unsqueeze(1))

                # 清空之前的梯度
                self.optimizer.zero_grad()
                # 反向传播损失
                loss.backward()
                # 更新参数
                self.optimizer.step()

                if self.lr_scheduler:
                    if self.lr_scheduler.__module__ == lr_scheduler.__name__:
                        # Using PyTorch In-Built scheduler
                        self.lr_scheduler.step()
                    else:
                        # Using custom defined scheduler
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.lr_scheduler(steps)

                total_loss += loss.item()
                steps += 1

            end_time = time.time()
            epoch_time = end_time - start_time

            if printTrainAcc:
                model_predictions, true_labels = predict_confusion(self.model, train_iter, device=self.device)
                for i in range(len(model_predictions)):
                    if model_predictions[i] < threshold:  # threshold
                        model_predictions[i] = 0
                    else:
                        model_predictions[i] = 1
                y_hat = model_predictions
                acc1 = accuracy_score(true_labels, y_hat)
            else:
                acc1 = 0
            if printTestAcc:
                model_predictions, true_labels = predict_confusion(self.model, test_iter, device=self.device)
                for i in range(len(model_predictions)):
                    if model_predictions[i] < threshold:  # threshold
                        model_predictions[i] = 0
                    else:
                        model_predictions[i] = 1
                y_hat = model_predictions
                acc2 = accuracy_score(true_labels, y_hat)
            else:
                acc2 = 0
            print(f'Model {model_num + 1}|Epoch:{epoch:003} | Time:{epoch_time:.2f}s')
            print(
                f'Train loss:{total_loss / len(train_iter)} ')
            # print(f'Train acc:{acc1}')
            print(f'Test acc:{acc2}')
            # 计算每个epoch和step的模型输出特征
            history1.log((epoch, steps),
                         train_loss=total_loss / len(train_iter),  # 训练集损失
                         train_acc=acc1,  #
                         test_acc=acc2)  # 验证集精度
            # 可视化网络训练的过程
            with canvas1:
                canvas1.draw_plot(history1["train_loss"])
                canvas1.draw_plot(history1["train_acc"])
                canvas1.draw_plot(history1["test_acc"])

            train_loss = total_loss / len(train_iter)
            if train_loss < best_loss:
                torch.save(self.model.state_dict(), best_model)
                best_loss = train_loss
                best_loss_acc = acc1
                bestlos_epoch = epoch


            if (best_loss < train_loss) and (epoch - bestlos_epoch >= early_stop):
                break



        self.model.load_state_dict(torch.load(best_model))
        os.remove(best_model)
        print("best_loss = " + str(best_loss))
        print("best_loss_acc = " + str(best_loss_acc))

        canvas1.save('./save_img/train_test_' + str(model_num + 1) + modelname + '.pdf')




class DataTrain_confusion_KD:
    def __init__(self, Student_model, Teacher_model, optimizer, criterion, temp=1, scheduler=None, device="cuda"):
        self.Student_model = Student_model.to(device)
        self.Teacher_model = Teacher_model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_scheduler = scheduler
        self.device = device
        self.temp = temp
        self.criterionCEL = torch.nn.BCEWithLogitsLoss()

    def distillation(self, y, labels, teacher_scores, temp, alpha):
        return nn.KLDivLoss()(F.log_softmax(y / temp, dim=1), F.softmax(teacher_scores / temp, dim=1)) * (
                temp * temp * 2.0 * alpha) + F.cross_entropy(y, labels) * (1. - alpha)

    def train_step_KD(self, train_dataset_s, test_dataset_s, train_dataset_t, modelname, epochs=None, model_num=0, early_stop = 10000, threshold = 0.5,printTrainAcc = False, printTestAcc = True):
        steps = 1
        best_loss = 100000.
        best_loss_acc = 0.
        bestlos_epoch = 0
        PATH = os.getcwd()
        best_model = os.path.join(PATH, 'saved_models', modelname + 'best.pth')

        history1 = hl.History()
        canvas1 = hl.Canvas()
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            total_loss = 0
            alpha = 0.7
            for (train_dna_s, train_protein_s, train_label_s), (train_dna_t, train_protein_t, _) in zip(train_dataset_s, train_dataset_t):
                # 使数据与模型在同一设备中
                train_dna_s, train_protein_s, train_label_s = train_dna_s.to(self.device), train_protein_s.to(self.device), train_label_s.to(
                    self.device)
                train_dna_t, train_protein_t = train_dna_t.to(self.device), train_protein_t.to(
                    self.device)
                self.Student_model.train()  # 进入训练模式
                # 教师模型预测
                with torch.no_grad():
                    out_t = self.Teacher_model(train_dna_t, train_protein_t)
                    # soft_label = nn.Sigmoid()(out_t.detach() / self.temp)

                _, out_s = self.Student_model.cls(train_dna_s, train_protein_s)


                # 计算损失
                # unsqueeze插入一个维度
                # loss =  self.distillation(out_s, train_label_s, out_t, self.temp, alpha)
                kd_loss = F.kl_div(
                F.log_softmax(out_s / self.temp, dim=1),
                F.log_softmax(out_t / self.temp, dim=1),
                reduction='sum',
                log_target=True
                    ) * (self.temp * self.temp) / out_s.numel()
                # kd_loss = self.criterionCEL(out_s / self.temp, soft_label.detach())
                stu_loss = self.criterion(out_s, train_label_s.float().unsqueeze(1))
                loss = stu_loss + kd_loss * alpha

                # 清空之前的梯度
                self.optimizer.zero_grad()
                # 反向传播损失
                loss.backward()
                # 更新参数
                self.optimizer.step()

                if self.lr_scheduler:
                    if self.lr_scheduler.__module__ == lr_scheduler.__name__:
                        # Using PyTorch In-Built scheduler
                        self.lr_scheduler.step()
                    else:
                        # Using custom defined scheduler
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.lr_scheduler(steps)

                total_loss += loss.item()
                steps += 1

            end_time = time.time()
            epoch_time = end_time - start_time

            if printTrainAcc:
                model_predictions, true_labels = predict_confusion(self.Student_model, train_dataset_s, device=self.device)
                for i in range(len(model_predictions)):
                    if model_predictions[i] < threshold:  # threshold
                        model_predictions[i] = 0
                    else:
                        model_predictions[i] = 1
                y_hat = model_predictions
                acc1 = accuracy_score(true_labels, y_hat)
            else:
                acc1 = 0
            if printTestAcc:
                model_predictions, true_labels = predict_confusion(self.Student_model, test_dataset_s, device=self.device)
                for i in range(len(model_predictions)):
                    if model_predictions[i] < threshold:  # threshold
                        model_predictions[i] = 0
                    else:
                        model_predictions[i] = 1
                y_hat = model_predictions
                acc2 = accuracy_score(true_labels, y_hat)
                test_score = estimate.evaluate(model_predictions, true_labels, 0.5)
                print(test_score)
            else:
                acc2 = 0
            print(f'Model {model_num + 1}|Epoch:{epoch:003} | Time:{epoch_time:.2f}s')
            print(
                f'Train loss:{total_loss / len(train_dataset_s)} ')
            # print(f'Train acc:{acc1}')
            print(f'Test acc:{acc2}')
            # 计算每个epoch和step的模型输出特征
            history1.log((epoch, steps),
                         train_loss=total_loss/ len(train_dataset_s),  # 训练集损失
                         train_acc=acc1,  #
                         test_acc=acc2)  # 验证集精度
            # 可视化网络训练的过程
            with canvas1:
                canvas1.draw_plot(history1["train_loss"])
                canvas1.draw_plot(history1["train_acc"])
                canvas1.draw_plot(history1["test_acc"])

            train_loss = total_loss / len(train_dataset_s)
            if epoch == 58:
                PATH = os.getcwd()
                torch.save(self.Student_model.state_dict(), os.path.join(PATH, 'saved_models', modelname + 'my_best.pth') )
            if train_loss < best_loss:
                torch.save(self.Student_model.state_dict(), best_model)
                best_loss = train_loss
                best_loss_acc = acc1
                bestlos_epoch = epoch


            if (best_loss < train_loss) and (epoch - bestlos_epoch >= early_stop):
                break



        self.Student_model.load_state_dict(torch.load(best_model))
        os.remove(best_model)
        print("best_loss = " + str(best_loss))
        print("best_loss_acc = " + str(best_loss_acc))

        canvas1.save('./save_img/train_test_' + str(model_num + 1) + modelname + '.pdf')


# model_predictions, true_labels
def predict(model, data, device="cuda"):
    # 模型预测
    model.to(device)
    model.eval()  # 进入评估模式
    predictions = []
    labels = []

    with torch.no_grad():  # 取消梯度反向传播
        for x, y in data:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)

            score = model.cls(x)
            label = torch.sigmoid(score)  # 将模型预测值映射至0-1之间
            predictions.extend(label.tolist())
            labels.extend(y.tolist())

    return np.array(predictions), np.array(labels)



def predict_confusion(model, data, device="cuda"):
    # 模型预测
    model.to(device)
    model.eval()  # 进入评估模式
    predictions = []
    labels = []

    with torch.no_grad():  # 取消梯度反向传播
        for x_dna, x_protein, y in data:
            x_dna = x_dna.to(device)
            x_protein = x_protein.to(device)
            y = y.to(device).unsqueeze(1)
            score = model.cls(x_dna,  x_protein)
            label = torch.sigmoid(score)  # 将模型预测值映射至0-1之间
            predictions.extend(label.tolist())
            labels.extend(y.tolist())

    return np.array(predictions), np.array(labels)

def predict_confusion_DA(model, data, device="cuda"):
    # 模型预测
    model.to(device)
    model.eval()  # 进入评估模式
    predictions = []
    labels = []

    with torch.no_grad():  # 取消梯度反向传播
        for x_dna, x_protein, y in data:
            x_dna = x_dna.to(device)
            x_protein = x_protein.to(device)
            y = y.to(device).unsqueeze(1)
            score = model(x_dna,  x_protein)
            label = torch.sigmoid(score)  # 将模型预测值映射至0-1之间
            predictions.extend(label.tolist())
            labels.extend(y.tolist())

    return np.array(predictions), np.array(labels)

def predict_confusion_cap(model, data, device="cuda"):
    # 模型预测
    model.to(device)
    model.eval()  # 进入评估模式
    predictions = []
    labels = []

    with torch.no_grad():  # 取消梯度反向传播
        for x_dna, x_protein, y , cap_fea in data:
            x_dna = x_dna.to(device)
            x_protein = x_protein.to(device)
            cap_fea = cap_fea.to(device)
            y = y.to(device).unsqueeze(1)

            score = model.cls(x_dna,  x_protein, cap_fea)
            label = torch.sigmoid(score)  # 将模型预测值映射至0-1之间
            predictions.extend(label.tolist())
            labels.extend(y.tolist())

    return np.array(predictions), np.array(labels)

def predict_confusion_pad(model, data, device="cuda"):
    # 模型预测
    model.to(device)
    model.eval()  # 进入评估模式
    predictions = []
    labels = []

    with torch.no_grad():  # 取消梯度反向传播
        for x_dna, x_protein, y, protein_len in data:
            x_dna = x_dna.to(device)
            x_protein = x_protein.to(device)
            protein_len = protein_len.to(device)
            y = y.to(device).unsqueeze(1)

            score = model.cls(x_dna,  x_protein, protein_len)
            label = torch.sigmoid(score)  # 将模型预测值映射至0-1之间
            predictions.extend(label.tolist())
            labels.extend(y.tolist())

    return np.array(predictions), np.array(labels)


def get_linear_schedule_with_warmup(optimizer_, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer_, lr_lambda, last_epoch)


class CosineScheduler:
    # 退化学习率
    def __init__(self, max_update, base_lr=0.01, final_lr=0, warmup_steps=0, warmup_begin_lr=0):
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps

    def get_warmup_lr(self, epoch):
        increase = (self.base_lr_orig - self.warmup_begin_lr) * float(epoch - 1) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase

    def __call__(self, epoch):
        if epoch < self.warmup_steps:
            return self.get_warmup_lr(epoch)
        if epoch <= self.max_update:
            self.base_lr = self.final_lr + (self.base_lr_orig - self.final_lr) * \
                           (1 + math.cos(math.pi * (epoch - 1 - self.warmup_steps) / self.max_steps)) / 2
        return self.base_lr
