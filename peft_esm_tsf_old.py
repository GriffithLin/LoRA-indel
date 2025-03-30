import os
from transformers import EsmTokenizer, EsmForMaskedLM, EsmModel, EsmForSequenceClassification
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
import esm, torch
from esm import FastaBatchedDataset
import deepspeed

from utils_esm import get_config, print_trainable_params, BatchConverter, get_avaliable_memory, Dataset_encoded_protein
from train_FT import FT_Train, predict_DS
from models.models_FT import FinetuneESM
from train import predict, CosineScheduler

load_dir = "/data1/linming/esm2_t33_650M_UR50D"
model_name = "facebook/esm2_t33_650M_UR50D"
# load_dir = "/data1/linming/esm2_t6_8M_UR50D"
# model_name = "facebook/esm2_t6_8M_UR50D"
train_data_path = "/data1/linming/DPPred-indel/dataCenter/train.fasta"
test_data_path = "/data1/linming/DPPred-indel/dataCenter/test.fasta"
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"

def train(args):
    # transformer 风格
    # get_avaliable_memory("cuda:0")
    # get_avaliable_memory("cuda:1")
    if load_dir is None:
        model = EsmForSequenceClassification.from_pretrained(model_name, num_labels=1)
        for name, param in model.named_parameters():
            if 'contact_head.regression' in name:
                print(param)
                print("param.requires_grad = False")
                param.requires_grad = False
    else:
        model = EsmForSequenceClassification.from_pretrained(load_dir, num_labels=1)
    # model = EsmModel.from_pretrained(model_name)
    tokenizer = EsmTokenizer.from_pretrained(load_dir)
    # _, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = BatchConverter(tokenizer)
    train_dataset = FastaBatchedDataset.from_file(train_data_path)
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, collate_fn=batch_converter, shuffle=True, batch_size = 2
    )
    test_dataset = FastaBatchedDataset.from_file(test_data_path)
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, collate_fn=batch_converter, shuffle=False, batch_size = 4
    )
    # for batch_idx, (labels, toks) in enumerate(test_data_loader):
    #     print(labels)
    #     print("toks:")
    #     print(toks)
    # return
    
    model = FinetuneESM(model, args.dropout, args.output_size)
    
    args.lora_modules = ["query", "key","value","dense"]
    lora_config = LoraConfig(r=args.lora_rank,
                        lora_alpha=32,
                        target_modules=args.lora_modules,
                        lora_dropout=0.1,
                        bias='none')
    model = get_peft_model(model, lora_config)
    
    
    print_trainable_params(model)
    # for name, param in model.named_parameters():
    #     print(param.requires_grad)
    #     print(name)
    get_avaliable_memory("cuda:0")
    get_avaliable_memory("cuda:1")
    
    if args.USE_DeepSpeed:
        # 设置设备
        args.device = torch.device(f'cuda:{args.local_rank}' if torch.cuda.is_available() else 'cpu')

        # 初始化 DeepSpeed
        # optimizer 用于卸载离线优化器参数，当优化器同时具有cpu和gpu实现时，可以不输入
        model,  optimizer, _, _ = deepspeed.initialize(
            args=args,
            model=model,
            # optimizer = optimizer,
            model_parameters=model.parameters(),
            # training_data=train_dataset  
        )
    else:
        optimizer = torch.optim.SGD(list(model.parameters()) , lr=args.lr, momentum=0.9,
                                    weight_decay=0.001, nesterov=True)


    criterion = torch.nn.BCEWithLogitsLoss()
    
    lr_scheduler = CosineScheduler(10000, base_lr=args.lr, warmup_steps=500
                                        ) 
    get_avaliable_memory("cuda:0")
    get_avaliable_memory("cuda:1")
    Train = FT_Train(model, optimizer, criterion, lr_scheduler, device = args.device, USE_DeepSpeed = args.USE_DeepSpeed)
    
    Train.train_step(train_data_loader, test_data_loader, args.model_name, args.epoch)
    
    PATH = os.getcwd()
    each_model = os.path.join(PATH, 'saved_models', args.model_name + '.pth')
    torch.save(model.state_dict(), each_model)
    
    
    
if __name__ == '__main__':
    args = get_config()
    print(args)
    
    # train(args)
    
    # test_dataset = FastaBatchedDataset.from_file(test_data_path)
    # test_data_loader = torch.utils.data.DataLoader(
    #     test_dataset, collate_fn=batch_converter, shuffle=False, batch_size = 2
    # )
    # model_path = os.path.join(PATH, 'saved_models', args.model_name + '.pth')
    # model.load_state_dict(torch.load(model_path))
    # predict_DS(model, test_data_loader)
    