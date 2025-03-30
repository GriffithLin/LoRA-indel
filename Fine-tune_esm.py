import os
from transformers import EsmTokenizer, EsmForMaskedLM
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
import esm, torch
from esm import FastaBatchedDataset
import deepspeed

from utils_esm import get_config, print_trainable_params
from train_FT import FT_Train
from models.models_FT import FinetuneESM
from train import predict, CosineScheduler

load_dir = "/data1/linming/esm2_t33_650M_UR50D"
model_name = "facebook/esm2_t33_650M_UR50D"
train_data_path = "/data3/linming/DPPred-indel/dataCenter/train.fasta"
test_data_path = "/data3/linming/DPPred-indel/dataCenter/test.fasta"

os.environ['CUDA_VISIBLE_DEVICES'] = "1"

def train(args):
    # transformer 风格
    # if load_dir is None:
    #     model = EsmForMaskedLM.from_pretrained(model_name)
    #     for name, param in model.named_parameters():
    #         if 'contact_head.regression' in name:
    #             param.requires_grad = False
    # else:
    #     model = EsmForMaskedLM.from_pretrained(load_dir)
    # tokenizer = EsmTokenizer.from_pretrained(load_dir)

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    train_dataset = FastaBatchedDataset.from_file(train_data_path)
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, collate_fn=batch_converter, shuffle=True, batch_size = 32
    )
    test_dataset = FastaBatchedDataset.from_file(test_data_path)
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, collate_fn=batch_converter, shuffle=False, batch_size = 32
    )
     
    lora_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "dense",
        "fc1",
        "fc2",
        "out_proj"
    ]
    lora_config = LoraConfig(r=args.lora_rank,
                        lora_alpha=32,
                        target_modules=lora_modules,
                        lora_dropout=0.1,
                        bias='none')
    model = get_peft_model(model, lora_config)
    
    model = FinetuneESM(model, args.dropout, args.output_size)
    
    print_trainable_params(model)

    # 设置设备
    # device = torch.device(f'cuda:{args.local_rank}' if torch.cuda.is_available() else 'cpu')

    # # 初始化 DeepSpeed
    # model, _, _, _ = deepspeed.initialize(
    #     args=args,
    #     model=model,
    #     model_parameters=model.parameters(),
    #     training_data=train_data_loader  
    # )
    


    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(list(model.parameters()) , lr=args.lr, momentum=0.9,
                                        weight_decay=0.001, nesterov=True)
    lr_scheduler = CosineScheduler(10000, base_lr=args.lr, warmup_steps=500
                                        ) 
    
    Train = FT_Train(model, optimizer, criterion, lr_scheduler, padding_idx = alphabet.padding_idx)
    Train.train_step(train_data_loader, test_data_loader, args.model_name, args.epoch)
    
    PATH = os.getcwd()
    each_model = os.path.join(PATH, 'saved_models', args.model_name + '.pth')
    torch.save(model.state_dict(), each_model)
    
    
if __name__ == '__main__':
    args = get_config()
    print(args)
    
    train(args)
    