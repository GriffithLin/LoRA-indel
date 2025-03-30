import argparse
import deepspeed
from typing import Sequence, Tuple, List, Union
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
#直接调用返回当前可用内存/显存
import torch
import pynvml
import psutil

def get_avaliable_memory(device):

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)        # 0表示第一块显卡
    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
    ava_mem=round(meminfo.free/1024**2)
    print(device)
    print('current available video memory is' +' : '+ str(round(meminfo.free/1024**2)) +' MIB')

    mem = psutil.virtual_memory()
    print('current available memory is' +' : '+ str(round(mem.used/1024**2)) +' MIB')
    ava_mem=round(mem.used/1024**2)

    return ava_mem

def print_trainable_params(model):
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            # print(f"Name: {name}, Shape: {param.shape}, Numel: {param.numel()}")
            trainable_params += param.numel()
    print(f'Trainable params: {trainable_params} ({100 * trainable_params / all_param:.2f}%)')
    print(f'All params: {all_param}')

# def print_trainable_params_ds(model):
#     trainable_params = 0
#     all_param = 0
#     for name, param in model.named_parameters():
#         all_param += param.numel()
#         if param.requires_grad:
#             deepspeed.utils.logging.info(name)
#             trainable_params += param.numel()
#     deepspeed.utils.logging.info(f'Trainable params: {trainable_params} ({100 * trainable_params / all_param:.2f}%)')
#     deepspeed.utils.logging.info(f'All params: {all_param}')

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

class BatchConverter_tsf(object):
    """Callable to convert an unprocessed (labels + strings) batch to a
    processed (labels + tensor) batch.
    """

    def __init__(self, tokenizer, truncation_seq_length: int = None):
        # self.alphabet = alphabet
        self.truncation_seq_length = truncation_seq_length
        self.tokenizer = tokenizer
        self.max_length = 1024
    def __call__(self, raw_batch: Sequence[Tuple[str, str]]):
        # RoBERTa uses an eos token, while ESM-1 does not.
        batch_size = len(raw_batch)
        batch_labels, seq_str_list = zip(*raw_batch)
        seq_encode_list = self.tokenizer(seq_str_list, padding= 'max_length', truncation = True, max_length=self.max_length,
                                         return_tensors="pt")
        labels = []

        for i, (label) in enumerate(
            zip(batch_labels)
        ):
            labels.append(float(label[-1][-1]))
        seq_encode_list["labels"] = torch.tensor(labels)
        # print(raw_batch)
        # print(seq_encode_list)
        return seq_encode_list
                    
class BatchConverter(object):
    """Callable to convert an unprocessed (labels + strings) batch to a
    processed (labels + tensor) batch.
    """

    def __init__(self, tokenizer, truncation_seq_length: int = None):
        # self.alphabet = alphabet
        self.truncation_seq_length = truncation_seq_length
        self.tokenizer = tokenizer
        self.max_length = 1024

    def cover(self, raw_batch: Sequence[Tuple[str, str]]):
        # print(raw_batch)
        # print(type(raw_batch))
        batch_label, seq_str_list = raw_batch
        
        seq_encode_list = self.tokenizer(seq_str_list, padding= 'max_length', truncation = True, max_length=self.max_length,
                                         return_tensors="pt")
        # 'max_length'
        labels = []
        # strs = []

        labels.append(int(batch_label[-1]))

        # print("seq_encode_list:")
        # print(seq_encode_list)
        return torch.tensor(labels), seq_encode_list
    
    def __call__(self, raw_batch: Sequence[Tuple[str, str]]):
        # print(raw_batch)
        # print(type(raw_batch))
        # RoBERTa uses an eos token, while ESM-1 does not.
        batch_size = len(raw_batch)
        batch_labels, seq_str_list = zip(*raw_batch)
        
        seq_encode_list = self.tokenizer(seq_str_list, padding= 'max_length', truncation = True, max_length=self.max_length,
                                         return_tensors="pt")
        labels = []

        for i, (label) in enumerate(
            zip(batch_labels)
        ):
            labels.append(int(label[-1][-1]))
        #     print("label:")
        #     print(label[-1])
        # print("seq_encode_list:")
        # print(seq_encode_list["input_ids"])
        # seq_encode_list["labels"] = labels
        return torch.tensor(labels), seq_encode_list
    
class Dataset_encoded_protein(Dataset):
    def __init__(self, labels, seq_encode_list, attention_mask_list):
        self.attention_mask_list = attention_mask_list
        self.seq_encode_list = seq_encode_list
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # print(idx)
        return (self.labels[idx]), (self.seq_encode_list[idx]), (self.attention_mask_list[idx])

def CoverDataset_with_batchCover(train_dataset, batch_converter):
    toks_input_ids = []
    toks_attention_mask = []
    labels = []
    for i in range(len(train_dataset)):
        processed_labels, seq_toks = batch_converter.cover(train_dataset[i])
        labels.extend(processed_labels.tolist())
        # print(seq_toks)
        toks_input_ids.extend(seq_toks["input_ids"])
        toks_attention_mask.extend(seq_toks["attention_mask"])

    return Dataset_encoded_protein(labels, toks_input_ids, toks_attention_mask)

# def CoverDataset_with_batchCover(train_dataset, batch_converter):
#     toks = []
#     labels = []
#     for i in range(len(train_dataset)):
#         processed_labels, seq_toks = batch_converter.cover(train_dataset[i])
#         labels.extend(processed_labels.tolist())
#         # print(seq_toks)
#         toks.extend(seq_toks)
#     # print("len(train_dataset)+++++++:", len(train_dataset))
#     print(len(toks))
#     print(toks[4616]["input_ids"])
#     # toks = {}
#     # toks["input_ids"] = toks_input_ids
#     # toks["attention_mask"] = toks_attention_mask
#     return Dataset_encoded_protein(labels, toks)


from transformers import TrainerCallback

class BestAucCallback(TrainerCallback):
    def __init__(self):
        self.best_auc = 0.0

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        # 更新最佳 AUC 值
        if "eval_AUC" in metrics and metrics["eval_AUC"] > self.best_auc:
            self.best_auc = metrics["eval_AUC"]
        print(f"Current best eval AUC: {self.best_auc}")
        
def get_config():
    parser = argparse.ArgumentParser(description='esm2 using DeepSpeed and fine tune')
    parser = deepspeed.add_config_arguments(parser)
    # train 
    parser.add_argument('--val_idx', type=int, default=0, help='index of cross val')
    parser.add_argument('--epoch', type=int, default=50, help='Number of training epoch')
    parser.add_argument('-batch_size', type=int, default=64, help='Input batch size for training')
    parser.add_argument('-lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('-dropout', type=float, default=0.6)
    parser.add_argument('-per_device_trai_nbatch_size', type=int, default=2)
        # deepspeed
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank passed from distributed launcher')
    parser.add_argument('--device', type=int, default=0, help='device')
    parser.add_argument('-ds', '--USE_DeepSpeed', action='store_true', default=False, help='USE_DeepSpeed')
        #lora
    parser.add_argument('-lora_modules', nargs='+', type=str, default=["query", "key","value","dense", "classifier", "pre_classifier"], help='')
    parser.add_argument('-lora_rank', type=int, default=1, help='lora_rank')
    # lora_alpha  lora_dropout  bias
    # data
    parser.add_argument('-output_size', type=int, default=1,
                    help='class to predict')
    #模型参数
    parser.add_argument('-model_name', type=str, default='finetuneESM',
                    help='Name of the model')
    
    config = parser.parse_args()
    return config