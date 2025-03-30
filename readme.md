# DPPred-indel
## install
This experiment involves two environments: DPPred-indel and DNABERT. 
The DNABERT environment is used for DNA feature extraction, while the DPPred-indel environment is used for protein feature extraction and running DPPred-indel.
### ①create dnabert environment
####  Create and activate a new virtual environment

```
conda create -n dnabert python=3.6
conda activate dnabert
```

#### Install the package and other requirements

(Required)

```
conda install pytorch torchvision cudatoolkit=12.2 -c pytorch == 1.8.0
cd DNABERT
python3 -m pip install --editable .
cd examples
python3 -m pip install -r requirements.txt
```


### ②create DPPred-indel environment
####  Create and activate a new virtual environment
```
conda create -n DPPred-indel python=3.7
conda activate DPPred-indel
```

#### Install the package and other requirements

(Required)

```
python3 -m pip install -r requirements.txt
```


## Data Preparation
#### Protein Feature Preparation

For details on extracting protein features, refer to the readme of ./esm.
you can get train.npy
#### DNA Feature Preparation

For details on extracting DNA features, refer to the readme of ./DNABERT.

you can get train_dna.npy

## Training and test DPPred-indel model
```shell
python DPPred-indel.py
```

## Related Files
| FILE NAME         | DESCRIPTION                                                                             |
|:------------------|:----------------------------------------------------------------------------------------|
| DPPred-indel.py           | the main file of DPPred-indel predictor |
| train.py          | train model                                                                             |
| models/model.py          | model construction                                                                      |
| my_util.py           | utils used to build models                                                              |
| loss_functions.py | loss functions used to train models                                                     |
| dataCenter           | data                                                                                    |
| result            | Models and results preserved during training.                                           |
| saved_models              | Saved model weight                                                                           |




# DPPred-Cindel

##  create DPPred-Cindel environment
DPPred-Cindel use the same environment with DPPred-indel
####  Create and activate a new virtual environment
```
conda create -n DPPred-indel python=3.7
conda activate DPPred-indel
```

#### Install the package and other requirements

(Required)

```
python3 -m pip install -r requirements.txt
```

## Data Preparation
#### Protein Feature Preparation for KD

For details on extracting protein features, refer to the readme of esm.
#### DNA Feature Preparation for KD

For details on extracting DNA features, refer to the readme of DNABERT.

## extra input data for student model and the implementation  of DA 
| dataset            | pos data | neg data | total  |
|-------------------|-------|-------|------|
| Ds-train-KD      | 5812  | 5302  | 11114 |
explanation:train dataset of DPPred-indel have had the target domain data removed.
protein seq：source_train_index.csv 
DNA k-mer seq：NotContext/5/225/source_train/dev.tsv
| Ds-testdata-KD      | 869   | 869   | 1738 |
explanation:the test dataset 1 of DPPred-indel.
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

