import os
from transformers import EsmTokenizer, EsmForMaskedLM, EsmModel, EsmForSequenceClassification, TrainingArguments, Trainer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, IA3Model, IA3Config
import esm, torch
from esm import FastaBatchedDataset
import deepspeed
import estimate, gc
import pandas as pd
import numpy as np

from utils_esm import BestAucCallback, get_config, print_trainable_params, BatchConverter_tsf, get_avaliable_memory, Dataset_encoded_protein, CoverDataset_with_batchCover, save_results
# from utils_esm import print_trainable_params_ds
from train_FT import  FT_Train_DS, predict_DS
from models.models_FT import FinetuneESM, FinetuneESM_DS
from train import predict, CosineScheduler
# from my_utils import save_results
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset
os.environ["MASTER_PORT"] = "29503" 


train_data_path = "dataCenter/train.fasta"
test_data_path = "dataCenter/test.fasta"
train_vcf_path = "dataCenter/train.vcf"
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"
num_gpu = 2

def print_tensor_info(batch):
    print(f"Input IDs dtype: {batch['input_ids'].dtype}")
    print(f"Labels dtype: {batch['labels'].dtype}")

def print_dataset(train_data_loader):
    for batch_idx, input in enumerate(train_data_loader):
        # 获取输入数据和标签
        input_ids = input["input_ids"]
        attention_mask = input["attention_mask"]
        labels = input["labels"]
        if batch_idx == 10:
            break
            # # 将输入传递给模型
        # output = model(input_ids=input_ids, attention_mask=attention_mask, labels = labels)
        # print(output.loss.dtype)# float32
        # print(labels.dtype)#int64
        # break
    return 

def predict(args):
    from peft import PeftModel, PeftConfig
    tokenizer = AutoTokenizer.from_pretrained(load_dir)
    batch_converter = BatchConverter_tsf(tokenizer)

    test_dataset = FastaBatchedDataset.from_file(test_data_path)
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset, collate_fn=batch_converter, shuffle=True, batch_size = 32
    )
    
    peft_model_id = args.model_name
    # 获取当前进程的 rank
    rank = int(os.environ.get("LOCAL_RANK", -1))
    config = PeftConfig.from_pretrained(peft_model_id)
    # model = AutoModelForSequenceClassification.from_pretrained('lora_650M_bf16_tsfbs64lora_rank8epoch20')
    model =  EsmForSequenceClassification.from_pretrained(config.base_model_name_or_path, num_labels=1, device_map={"":0})
    model = PeftModel.from_pretrained(model, peft_model_id, device_map={"":0})
    model.eval()
    
    # 假设你已经定义了一个 test_dataset
    training_args = TrainingArguments(
        output_dir='./results',
        per_device_eval_batch_size=8,  # 根据你的需要调整 batch size
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,  # 传入 tokenizer
    )
    predictions = trainer.predict(test_dataset)
    print(predictions)

def interface_predict(args, sequence = None):
    from peft import PeftModel, PeftConfig
    import matplotlib.pyplot as plt
    
    sequence = "MAPKVVVLLLLGILGTVL"
    # 打印注意力权重形状
    # model_dir = "/data3/linming/DPPred-indel/saved_models/LoRA_150M_bf16_tsfbs64lora_rank8epoch1val/"
    model_dir = "/data3/linming/DPPred-indel/saved_models/" + str(args.model_name[:-1])
    # 1. 加载基础模型（无 LoRA）
    base_model = EsmForSequenceClassification.from_pretrained(args.load_dir, num_labels=1)
    base_model.eval()

    tokenizer = EsmTokenizer.from_pretrained(args.load_dir)
    batch_converter = BatchConverter_tsf(tokenizer)
    dataset = FastaBatchedDataset.from_file(train_data_path)
    # 3. 定义推理函数
    def get_attention(model, sequence):
        inputs = tokenizer(
            sequence,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        )
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
        return outputs.attentions

    # 获取无 LoRA 的注意力图
    attention_map_no_lora = get_attention(base_model, sequence)

    for layer, attention in enumerate(attention_map_no_lora):
        print(f"Layer {layer}: {attention.shape}")
    # 2. 加载带 LoRA 的模型
    lora_model = PeftModel.from_pretrained(base_model, model_dir, torch_dtype=torch.bfloat16)
    lora_model.eval()
    
    # 获取有 LoRA 的注意力图
    attention_map_with_lora = get_attention(lora_model, sequence)

    # 确保层数不超过 attention_map 的层数
    num_layers = len(attention_map_no_lora)
    layers_to_show = list(range(num_layers - 5, num_layers)) if num_layers >= 5 else list(range(num_layers))
    
    # 创建子图
    fig, ax = plt.subplots(2, len(layers_to_show), figsize=(5 * num_layers, 4), sharey=True)

    
def finetune_single(train_dataset, val_set, test_dataset, batch_converter, args):
    rank = int(os.environ.get("LOCAL_RANK", -1))
    # transformer 风格
    if args.load_dir is None:
        model = EsmForSequenceClassification.from_pretrained(args.load_model_name, num_labels=1)
        for name, param in model.named_parameters():
            if 'contact_head.regression' in name:
                print(param)
                print("param.requires_grad = False")
                param.requires_grad = False
    else:
        model =  EsmForSequenceClassification.from_pretrained(args.load_dir, num_labels=1)
    
    if args.FT_method == "LoRA":
        # args.lora_modules = ["query", "key","value","intermediate.dense", "output.dense"]
        
        lora_config = LoraConfig(r=args.lora_rank,
                            lora_alpha=args.lora_alpha,
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
    elif args.FT_method == "IA3":
        ia3_config = IA3Config(
            peft_type="IA3",
            task_type="SEQ_2_SEQ_LM",
            target_modules=["key", "value" , "intermediate.dense", "output.dense"],
            feedforward_modules=["intermediate.dense", "output.dense"],
        ) 
        model.add_adapter(ia3_config)
    elif args.FT_method == "PromptTuning":
        # from peft import PromptTuningConfig, PromptTuningInit, get_peft_model

        # >>> config = {
        # ...     "peft_type": "PREFIX_TUNING",
        # ...     "task_type": "SEQ_CLS",
        # ...     "inference_mode": False,
        # ...     "num_virtual_tokens": 20,
        # ...     "token_dim": 768,
        # ...     "num_transformer_submodules": 1,
        # ...     "num_attention_heads": 12,
        # ...     "num_layers": 12,
        # ...     "encoder_hidden_size": 768,
        # ...     "prefix_projection": False,
        # ...     "postprocess_past_key_value_function": None,
        # ... }

        # # op_feature_list
        # # prompt_tuning_init_text = "Classify if the tweet is a complaint or no complaint.\n"
        # peft_config = PromptTuningConfig(
        #     task_type="SEQ_CLS",
        #     prompt_tuning_init=PromptTuningInit.RANDOM,
        #     num_virtual_tokens=50,
        #     # prompt_tuning_init_text=prompt_tuning_init_text,
        #     tokenizer_name_or_path=args.load_dir,
        # )

        from peft import PromptEncoderConfig, get_peft_model
        peft_config = PromptEncoderConfig(task_type="CAUSAL_LM", num_virtual_tokens=20, encoder_hidden_size=2560)
        model = get_peft_model(model, peft_config)
        # model.print_trainable_parameters()
        
    if rank == 0:
        for name, param in model.named_parameters():
            print(param.requires_grad)
            print(name)
        print_trainable_params(model)

    # 创建回调实例
    best_auc_callback = BestAucCallback()

    # 设置 训练参数 ，要放到模型加载之后
    if args.FT_method == "FT":
        deepspeed_config = "ds_config_tsf.json"
    else:
        deepspeed_config = "o1.json"
    
    output_dir = "saved_models"
    # gradient_checkpointing_enable
    training_args = TrainingArguments(
        data_seed = 42,
        output_dir=output_dir + args.model_name,
        per_device_train_batch_size=args.per_device_trai_nbatch_size,
        per_device_eval_batch_size=args.per_device_trai_nbatch_size,
        gradient_accumulation_steps=int(args.batch_size/(args.per_device_trai_nbatch_size * num_gpu)),
        eval_accumulation_steps=1, # 防止评估时导致OOM
        # bf16 = True,
        fp16 = True,
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
        eval_dataset=val_set,
        data_collator=batch_converter,
        compute_metrics=estimate.evaluate_tsf,
        # callbacks = [print_tensor_info],
        callbacks=[best_auc_callback],  # 添加回调
    )

    trainer.train()
    gc.collect()
    
    torch.cuda.empty_cache()
    # trainer.save_model(args.model_name)
    # trainer.model.save_pretrained(args.model_name)
    # tokenizer.save_pretrained(args.model_name)

    predictions = trainer.predict(test_dataset)
    if rank == 0:
        # 提取预测结果
        predictions_array = predictions.predictions  # 或者根据实际返回值提取

        # 保存预测结果到 .npy 文件
        np.save(args.model_name[:-1] + 'predictions.npy', predictions_array)

        filename = args.model_name[:-1] + f"predictions_val.txt"
        # 将预测结果写入文件
        with open(filename, 'a') as f:
            f.write(args.model_name + f" Predictions:\n")
            f.write(str(predictions))  # 将 predictions 转换为字符串并写入文件
        
        print(f"Predictions saved to {filename}")  # 打印保存的文件名

    if rank == 0:
        # 提取预测值和标签
        pred_values = predictions.predictions.flatten()  # 将预测值展平为一维数组
        labels = predictions.label_ids  # 真实标签

        # 将预测值和标签组合成表格
        results_df = pd.DataFrame({
            'Predictions': pred_values,
            'Labels': labels
        })

        # 添加评估指标到表格中
        metrics = predictions.metrics
        metrics["best_val_AUC"] = best_auc_callback.best_auc
        metrics_df = pd.DataFrame({
            'Model_Name': [args.model_name],  # 模型名称
            "best_val_AUC": [metrics["best_val_AUC"]], 
            'AUC': [metrics['test_AUC']],     # AUC 值
            'Recall': [metrics['test_recall']],  # 召回率
            'SPE': [metrics['test_SPE']],     # 特异性
            'PRE': [metrics['test_PRE']],     # 精确率
            'F1': [metrics['test_F1']],       # F1 分数
            'ACC': [metrics['test_ACC']]      # 准确率
        })

        # 保存预测结果和评估指标到文件
        # results_filename = args.model_name[:-1] + "_predictions_val.csv"
        metrics_filename = args.model_name[:-1] + "_metrics_val.csv"

        # results_df.to_csv(results_filename, index=False)

        # 定义固定文件名
        metrics_filename = "all_metrics_results.csv"

        # 检查文件是否存在，如果存在则追加，否则创建新文件
        if os.path.exists(metrics_filename):
            metrics_df.to_csv(metrics_filename, mode='a', header=False, index=False)
        else:
            metrics_df.to_csv(metrics_filename, index=False)

        # print(f"Predictions saved to {results_filename}")
        print(f"Metrics saved to {metrics_filename}")


# alpha_list = [2, 4, 64]
# ["query",  "key", "value"]
# lora_modules_list = [["key", "value"], ["query"], ["key"]]
# lora_name_list = ["KV", "Q", "K"]
def train(args):
    # 获取当前进程的 rank
    rank = int(os.environ.get("LOCAL_RANK", -1))

    tokenizer = EsmTokenizer.from_pretrained(args.load_dir)
    batch_converter = BatchConverter_tsf(tokenizer)
    dataset = FastaBatchedDataset.from_file(train_data_path)
    test_dataset = FastaBatchedDataset.from_file(test_data_path)
    tran_vcf = pd.read_csv(train_vcf_path, sep = "\t")
    train_label = tran_vcf["INFO"].apply(lambda x:float(x[-1]))
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    train_datasets, val_datasets = [], []       
    i = 0
    for train_index, val_index in cv.split(dataset, train_label):
        if i != args.val_idx:
            i = i + 1
            continue
        train_dataset = Subset(dataset, train_index.tolist())
        val_dataset = Subset(dataset, val_index.tolist())
        train_datasets.append(train_dataset)
        val_datasets.append(val_dataset) 

    model_name = args.model_name
    # 实际只做一次
    for i, (train_dataset, val_dataset) in enumerate(zip(train_datasets, val_datasets)):
        # for args.lora_modules, lora_name in zip(lora_modules_list, lora_name_list):
        # for args.lora_alpha in alpha_list:
        args.model_name = model_name  + "lora_alpha" + str(args.lora_alpha)+ "_lora_rank" + str(args.lora_rank)   + "val" + str(args.val_idx)  
        finetune_single(train_dataset, val_dataset, test_dataset, batch_converter, args)
        break
    

if __name__ == '__main__':
    args = get_config()
    args.epoch = 20
    # args.lr = 5e-4
    args.batch_size = 64
    args.lora_rank = 8
    args.lora_alpha = 2 * args.lora_rank
    # args.lora_alpha = 64
    # args.lr = 0.0005
    args.FT_method = "LoRA"
    args.param_num = "8M"
    if args.FT_method == "FT":
        args.learning_rate = 0.0005
    
    if args.param_num == "3B":
        args.load_dir = "/data2/liujie/linming/esm2_t36_3B_UR50D"
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
    elif args.param_num == "15B":
        args.load_dir = "/data2/liujie/linming/esm2_t48_15B_UR50D/"
    elif args.param_num == "35M":
        args.load_dir = "/data1/linming/esm2_t12_35M_UR50D/"

    # # ["query", "key","value","intermediate.dense", "output.dense"]
    args.lora_modules = ["query",  "key", "value"]
    args.model_name = args.FT_method + "_" + args.param_num + "_hf16_tsf" + "bs" + str(args.batch_size) + "epoch" + str(args.epoch) + "lr" + str(args.lr)
    args.per_device_trai_nbatch_size = 16


    print(args)
    train(args)
    
    # args.model_name = "KV" + args.model_name  + "lora_alpha" + str(args.lora_alpha)+ "_lora_rank" + str(args.lora_rank)   + "val0"   

    # args.batch_size = 1
    # interface_predict_seq(args)
    # interface_predict_seq_lora(args)
    # get_predict_lora(args) 
    
