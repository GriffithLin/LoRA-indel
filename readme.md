# LoRA-indel
## install

####  Create and activate a new virtual environment
```
conda create -n LoRA-indel python=3.7
conda activate LoRA-indel
```

#### Install the package and other requirements

(Required)

```
conda env create -f environment.yml
```


## Data Preparation
#### Protein Feature Preparation

For details on extracting protein features, refer to the readme of ./esm.
you can get train.npy
#### DNA Feature Preparation

For details on extracting DNA features, refer to the readme of ./DNABERT.

you can get train_dna.npy

## Training and test LoRA-indel model
```shell
python LoRA-indel.py
```

## Related Files
| FILE NAME         | DESCRIPTION                                                                             |
|:------------------|:----------------------------------------------------------------------------------------|
| LoRA-indel.py           | the main file of LoRA-indel predictor |
| train.py          | train model                                                                             |
| models/model.py          | model construction                                                                      |
| my_util.py           | utils used to build models                                                              |
| loss_functions.py | loss functions used to train models                                                     |
| dataCenter           | data                                                                                    |
| result            | Models and results preserved during training.                                           |
| saved_models              | Saved model weight                                                                           |



## Data Preparation
#### Protein Feature Preparation for KD

For details on extracting protein features, refer to the readme of esm.
#### DNA Feature Preparation for KD

For details on extracting DNA features, refer to the readme of DNABERT.

## extra input data for student model and the implementation  of DA 
| dataset            | pos data | neg data | total  |
|-------------------|-------|-------|------|
| Ds-train-KD      | 5812  | 5302  | 11114 |
explanation:train dataset of LoRA-indel have had the target domain data removed.
protein seq：source_train_index.csv 
DNA k-mer seq：NotContext/5/225/source_train/dev.tsv
| Ds-testdata-KD      | 869   | 869   | 1738 |
explanation:the test dataset 1 of LoRA-indel.
protein seq：test_protein.csv
DNA k-mer seq：NotContext/5/225/dev.tsv
| Ds-DA           | 6666  | 6153  | 12819 |
explanation:The source domain's training and test sets have had the target domain data removed.
protein seq：source_Data_index.csv
DNA k-mer seq：NotContext/5/225/source/train.tsv
| Target Domain train Set       | -     | -     | 5808 |
protein seq：target_index.csv
DNA k-mer seq：NotContext/5/225/DA/train.tsv
| Target Domain Test Set       | 200   | 182   | 382   |
protein seq：target_test_index.csv
DNA k-mer seq：NotContext/5/225/DA/dev.tsv

## Training and test DPPred-Cindel model

```shell
# train teachre model
python trainForKD.py
# train student model via knowledge distillation
python KD_MAIN_best.py
# Training and test DPPred-Cindel model
python DA_DANN_student.py
```


## related file
| FILE NAME         | DESCRIPTION                                                                             |
|:------------------|:----------------------------------------------------------------------------------------|
|trainForKD.py | train base teacher model|
|KD_MAIN_best.py | the main file of knowledge distillation|
|DA_DANN_student.py |  the main file of Domain adaption |

