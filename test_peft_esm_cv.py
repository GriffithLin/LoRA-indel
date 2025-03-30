import os
from transformers import EsmTokenizer, EsmForMaskedLM, EsmModel, EsmForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
import esm, torch
from esm import FastaBatchedDataset
import deepspeed
import estimate, gc

from utils_esm import get_config, print_trainable_params, BatchConverter_tsf, get_avaliable_memory, Dataset_encoded_protein, CoverDataset_with_batchCover, save_results
# from utils_esm import print_trainable_params_ds
from train_FT import  FT_Train_DS, predict_DS
from models.models_FT import FinetuneESM, FinetuneESM_DS
from train import predict, CosineScheduler
# from my_utils import save_results


# load_dir = "/data1/linming/esm2_t36_3B_UR50D"
# model_name = "facebook/esm2_t36_3B_UR50D"
load_dir = "/data1/linming/esm2_t33_650M_UR50D"
model_name = "facebook/esm2_t33_650M_UR50D"
# load_dir = "/data1/linming/esm2_t30_150M_UR50D"
# model_name = "facebook/esm2_t30_150M_UR50D"
# load_dir = "/data1/linming/esm2_t6_8M_UR50D"
# model_name = "facebook/esm2_t6_8M_UR50D"
train_data_path = "/data1/linming/DPPred-indel/dataCenter/train.fasta"
test_data_path = "/data1/linming/DPPred-indel/dataCenter/test.fasta"
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
num_gpu = 2

def print_tensor_info(batch):
    print(f"Input IDs dtype: {batch['input_ids'].dtype}")
    print(f"Labels dtype: {batch['labels'].dtype}")

def train(args):
    # 获取当前进程的 rank
    rank = int(os.environ.get("LOCAL_RANK", -1))

    tokenizer = EsmTokenizer.from_pretrained(args.load_dir)
    batch_converter = BatchConverter_tsf(tokenizer)
    train_dataset = FastaBatchedDataset.from_file(train_data_path)
    test_dataset = FastaBatchedDataset.from_file(test_data_path)
    
    deepspeed_config = "ds_config_tsf.json"
    output_dir = "saved_models"


    if args.load_dir is None:
        model = EsmForSequenceClassification.from_pretrained(model_name, num_labels=1)
        for name, param in model.named_parameters():
            if 'contact_head.regression' in name:
                print(param)
                print("param.requires_grad = False")
                param.requires_grad = False
    else:
        model =  EsmForSequenceClassification.from_pretrained(args.load_dir, num_labels=1)

    if args.FT_method == "LoRA":
        args.lora_modules = ["query", "key","value","intermediate.dense", "output.dense"]
        lora_config = LoraConfig(r=args.lora_rank,
                            lora_alpha=args.lora_rank * 2,
                            target_modules=args.lora_modules,
                            lora_dropout=0.1,
                            bias='none')
        #这种方式不兼容trainer
        # model = get_peft_model(model, lora_config)
        model.add_adapter(lora_config)
        # # model = FinetuneESM_DS(model, args.dropout, args.output_size)

        # 设置分类器梯度
        for name, param in model.named_parameters():
            if "model.classifier" in name:
                param.requires_grad = True
            if "classifier" in name:
                param.requires_grad = True
    elif args.FT_method == "MLP":
        print("args.FT_method == MLP")
        for name, param in model.named_parameters():
            param.requires_grad = False
            if "model.classifier" in name:
                param.requires_grad = True
            if "classifier" in name:
                param.requires_grad = True
    
    if rank == 0:
        for name, param in model.named_parameters():
            print(param.requires_grad)
            print(name)
        print_trainable_params(model)
    
    
        # gradient_checkpointing_enable
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=args.per_device_trai_nbatch_size,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=int(args.batch_size/(args.per_device_trai_nbatch_size * num_gpu)),
        eval_accumulation_steps=1, # 防止评估时导致OOM
        bf16 = True,
        # fp16 = True,
        learning_rate=args.lr,
        num_train_epochs=args.epoch,
        # logging & evaluation strategies
        logging_dir="logs",
        # logging_strategy="steps",
        # logging_steps=50, # 每50个step打印一次log
        # evaluation_strategy="steps",
        # eval_steps=500, # 每500个step进行一次评估
        # save_steps=500,
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        deepspeed=deepspeed_config,# deepspeed配置文件的位置
        lr_scheduler_type = "cosine",
        warmup_steps = 500,
        weight_decay = 0.1,
        adam_beta2 = 0.95,
        gradient_checkpointing = True,
        metric_for_best_model="AUC",
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        data_collator=batch_converter,
        compute_metrics=estimate.evaluate_tsf,
        # callbacks = [print_tensor_info],
    )

    trainer.train()
    gc.collect()
    torch.cuda.empty_cache()
    # trainer.save_model(args.model_name)
    trainer.model.save_pretrained(args.model_name)
    tokenizer.save_pretrained(args.model_name)

    predictions = trainer.predict(test_dataset)
    if rank == 0:
        print(predictions)
        
        # transformer 风格

if __name__ == '__main__':
    args = get_config()
    args.epoch = 20
    # args.lr = 5e-4
    args.batch_size = 64
    args.lora_rank = 8
    args.FT_method = "MLP"
    args.param_num = "8M"
    if args.param_num == "3B":
        args.load_dir = "/data1/linming/esm2_t36_3B_UR50D"
        args.load_model_name = "facebook/esm2_t36_3B_UR50D"
    elif args.param_num == "650M":
        args.load_dir = "/data1/linming/esm2_t33_650M_UR50D"
        args.load_model_name = "facebook/esm2_t33_650M_UR50D"
    elif args.param_num == "150M":
        args.load_dir = "/data1/linming/esm2_t30_150M_UR50D"
        args.load_model_name = "facebook/esm2_t30_150M_UR50D"
    elif args.param_num == "8M":
        args.load_dir = "/data1/linming/esm2_t6_8M_UR50D"
        args.load_model_name = "facebook/esm2_t6_8M_UR50D"
    args.model_name = args.FT_method + "_" + args.param_num + "_bf16_tsf" + "bs" + str(args.batch_size) + "lora_rank" + str(args.lora_rank) + "epoch" + str(args.epoch)
    args.per_device_trai_nbatch_size = 32


    print(args)
    train(args)
    # predict(args)
    
