import time
import torch
import math
import numpy as np
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import LambdaLR
from sklearn.metrics import accuracy_score, precision_recall_curve,auc
import os
import shutil
import hiddenlayer as hl
import torch.nn as nn
from train import predict, CosineScheduler
import estimate
# from models.model import MultipleKernelMaximumMeanDiscrepancy, JointMultipleKernelMaximumMeanDiscrepancy, \
#     DomainAdversarialLoss, ConditionalDomainAdversarialLoss
import sys
from my_util import ForeverDataIterator
import torch.nn.functional as F
from utils_esm import get_avaliable_memory

def predict(model, data, device="cuda"):
    # 模型预测
    model.to(device)
    model.eval()  # 进入评估模式
    predictions = []
    labels = []

    with torch.no_grad():  # 取消梯度反向传播
        for batch_idx, (y, toks) in enumerate(data):
            toks = toks.to(device)
            y = y.to(device).unsqueeze(1)
            score = model(toks["input_ids"], toks["attention_mask"])
            label = torch.sigmoid(score)  # 将模型预测值映射至0-1之间
            predictions.extend(label.tolist())
            labels.extend(y.tolist())

    return np.array(predictions), np.array(labels)

def predict_DS(model, data, device="cuda"):
    # 模型预测
    model.to(device)
    model.eval()  # 进入评估模式
    predictions = []
    labels = []

    with torch.no_grad():  # 取消梯度反向传播
        for batch_idx, (y, toks, attention_mask) in enumerate(data):
            toks = toks.to(device)
            attention_mask = attention_mask.to(device)
            y = y.to(device).unsqueeze(1)
            score = model(toks, attention_mask)
            label = torch.sigmoid(score)  # 将模型预测值映射至0-1之间
            predictions.extend(label.tolist())
            labels.extend(y.tolist())

    return np.array(predictions), np.array(labels)

class FT_Train_DS:
    def __init__(self, model, optimizer, criterion, scheduler=None, device="cuda", USE_DeepSpeed = False):
        self.USE_DeepSpeed = USE_DeepSpeed
        if not USE_DeepSpeed:
            self.model = model.to(device)
        else:
            self.model = model
            
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_scheduler = scheduler
        self.device = device
        # self.padding_idx = padding_idx
    
    def train_step(self, train_iter, test_iter, modelname, epochs=None, model_num=0, early_stop = 10000, threshold = 0.5):
        steps = 1
        best_loss = 100000.
        best_loss_acc = 0.
        bestlos_epoch = 0
        PATH = os.getcwd()
        best_model = os.path.join(PATH, 'saved_models', modelname + 'best.pth')

        print(type(train_iter))
        print(train_iter)
        history1 = hl.History()
        canvas1 = hl.Canvas()
        for epoch in range(1, epochs + 1):

            start_time = time.time()
            total_loss = 0
            # self.model.eval() 
            # with torch.no_grad(): 
            # print(next(iter(train_iter)))
            for batch_idx, (labels, toks, attention_masks)in enumerate(train_iter):
                # get_avaliable_memory("cuda:0")
                # get_avaliable_memory("cuda:1")
                # print(sys.getsizeof(batch_data))
                # print(sys.getsizeof(labels))
                # print(batch_data)
                # print(labels)
                if torch.cuda.is_available(): 
                    # print("to_cuda")
                    toks = toks.to(device=self.device)
                    attention_masks = attention_masks.to(device=self.device)
                    labels = labels.to(device=self.device)
                    
                # get_avaliable_memory("cuda:0")
                # get_avaliable_memory("cuda:1")
                self.model.train()

                preded = self.model(toks, attention_masks)
                # print("after train+++++++++++++")
                # get_avaliable_memory("cuda:0")
                # get_avaliable_memory("cuda:1")
                
                loss = self.criterion(preded, labels.float().unsqueeze(1))
                # print(loss.requires_grad, loss.grad_fn)
                # print(loss.item())
                self.optimizer.zero_grad()
                if self.USE_DeepSpeed :
                    self.model.backward(loss)
                else:
                    loss.backward()
                self.optimizer.step()
                if self.lr_scheduler:
                    if self.lr_scheduler.__module__ == self.lr_scheduler.__class__.__name__:
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
            
            model_predictions, true_labels = predict(self.model, test_iter, device=self.device)
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

            # 保留最佳模型
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
      
      

class FT_Train:
    def __init__(self, model, optimizer, criterion, scheduler=None, device="cuda", USE_DeepSpeed = False):
        self.USE_DeepSpeed = USE_DeepSpeed
        if not USE_DeepSpeed:
            self.model = model.to(device)
        else:
            self.model = model
            
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_scheduler = scheduler
        self.device = device
        # self.padding_idx = padding_idx
    
    def train_step(self, train_iter, test_iter, modelname, epochs=None, model_num=0, early_stop = 10000, threshold = 0.5):
        steps = 1
        best_loss = 100000.
        best_loss_acc = 0.
        bestlos_epoch = 0
        PATH = os.getcwd()
        best_model = os.path.join(PATH, 'saved_models', modelname + 'best.pth')

        print(type(train_iter))
        print(train_iter)
        history1 = hl.History()
        canvas1 = hl.Canvas()
        for epoch in range(1, epochs + 1):

            start_time = time.time()
            total_loss = 0
            # self.model.eval() 
            # with torch.no_grad(): 
            # print(next(iter(train_iter)))
            for batch_idx, (labels, batch_data)in enumerate(train_iter):
                # get_avaliable_memory("cuda:0")
                # get_avaliable_memory("cuda:1")
                # print(sys.getsizeof(batch_data))
                # print(sys.getsizeof(labels))
                # print(batch_data)
                # print(labels)
                if torch.cuda.is_available(): 
                    # print("to_cuda")
                    batch_data = batch_data.to(device=self.device)
                    labels = labels.to(device=self.device)
                    
                # get_avaliable_memory("cuda:0")
                # get_avaliable_memory("cuda:1")
                self.model.train()

                preded = self.model(batch_data)
                # print("after train+++++++++++++")
                # get_avaliable_memory("cuda:0")
                # get_avaliable_memory("cuda:1")
                
                loss = self.criterion(preded, labels.float().unsqueeze(1))
                # print(loss.requires_grad, loss.grad_fn)
                # print(loss.item())
                self.optimizer.zero_grad()
                if self.USE_DeepSpeed :
                    self.model.backward(loss)
                else:
                    loss.backward()
                self.optimizer.step()
                if self.lr_scheduler:
                    if self.lr_scheduler.__module__ == self.lr_scheduler.__class__.__name__:
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
            
            model_predictions, true_labels = predict(self.model, test_iter, device=self.device)
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

            # 保留最佳模型
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
        
class FT_Train_esm:
    def __init__(self, model, optimizer, criterion, scheduler=None, device="cuda", padding_idx = None):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_scheduler = scheduler
        self.device = device
        self.padding_idx = padding_idx
    
    def train_step(self, train_iter, test_iter, modelname, epochs=None, model_num=0, early_stop = 10000, threshold = 0.5, USE_DeepSpeed = False):
        steps = 1
        best_loss = 100000.
        best_loss_acc = 0.
        bestlos_epoch = 0
        PATH = os.getcwd()
        best_model = os.path.join(PATH, 'saved_models', modelname + 'best.pth')

        print(type(train_iter))
        print(train_iter)
        history1 = hl.History()
        canvas1 = hl.Canvas()
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            total_loss = 0
            self.model.eval() 
            # with torch.no_grad(): 
            for batch_idx, (labels, strs, toks) in enumerate(train_iter):
                batch_lens = (toks != self.padding_idx).sum(1)
                print(batch_lens)
                # print(strs)
                # print(toks.shape)
                
                if torch.cuda.is_available(): 
                    print("to_cuda")
                    toks = toks.to(device=self.device, non_blocking=True)
                # self.model.train()
                # preded = self.model(toks)
                # loss = self.criterion(preded.unsqueeze(1), labels.float().unsqueeze(1))
                
                # self.optimizer.zero_grad()
                # if USE_DeepSpeed:
                #     self.model.backward(loss)
                # else:
                #     loss.backward()
                # self.optimizer.step()
                # if self.lr_scheduler:
                #     if self.lr_scheduler.__module__ == self.lr_scheduler.__name__:
                #         # Using PyTorch In-Built scheduler
                #         self.lr_scheduler.step()
                #     else:
                #         # Using custom defined scheduler
                #         for param_group in self.optimizer.param_groups:
                #             param_group['lr'] = self.lr_scheduler(steps)
                total_loss += loss.item()
                steps += 1
            
            end_time = time.time()
            epoch_time = end_time - start_time
            
            model_predictions, true_labels = predict(self.model, test_iter, device=self.device)
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

            # 保留最佳模型
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