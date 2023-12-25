# SAFFECODE

SafeCode: A Framework for Learning Causal Features to Enhance DL-based Code Classification Robustness

## Requirement

conda env create -f safecode-environment.yml

## Preparing the Dataset

we use the same dataset as carrot : https://github.com/SEKE-Adversary/CARROT

**Use the pre-processed datasets**

1. Download the already pre-processed datasets -- [OJ](https://drive.google.com/drive/folders/1__SjuEKH8Sa_OYWhegiGE6Brbr1ObZrM?usp=sharing) and [CodeChef](https://drive.google.com/drive/folders/1ZEIb35PzfD2ojWr53Qa_myFRMVK7QI7f?usp=sharing).

2. Put the contents of OJ and CodeChef in the corresponding directories of `data` and `data_defect` respectively.

3. The pre-processed datasets for LSTM and BERT are all included in the directories now.

**Pre-process the datasets by yourself**

1. Download the raw datasets, *i.e.*, `oj.tar.gz` from [OJ](https://drive.google.com/drive/folders/1__SjuEKH8Sa_OYWhegiGE6Brbr1ObZrM?usp=sharing) and `codechef.zip` from [CodeChef](https://drive.google.com/drive/folders/1ZEIb35PzfD2ojWr53Qa_myFRMVK7QI7f?usp=sharing)

2. Put `oj.tar.gz` in the directory of `data` and `codechef.zip` in `data_defect`.

3. Run the following commands to build the OJ dataset for the DL models. The dataset format of CodeChef is almost identical to OJ, and the code can be reused.

```sh
> cd preprocess-lstm
> python3 main.py 
```

4. Copy `oj.pkl.gz`, `oj_uid.pkl.gz`, and `oj_inspos.pkl.gz` in the directory of `data`.


## Train
graphcodebert in train_bert
lstm in train_lstm

Training the safecode model is done in 'train_safecode' function

```
python run.py   --enhance_method SAFECODE  --task code_function --epochs 25 --early_stop 3 --domain_list origin,random_token  --attack_type token  
```

```sh
python run.py \
--do_train --do_renew --do_eval --do_attack  \
--enhance_method safecode --attack_type token --task code_defect \
--epochs 28  --early_stop 3   \
--domain_list origin,LSTM-safecode-token \
```


## Adversarial Attack

Run `python3 attacker.py` in each directory to attack the DL models.

*E.g.*, run the following commands to attack the graphcodebert model on OJ.

```sh
> cd train_lstm
> python3 attacker.py --model_dir FINETUNED/CODEBERT/MODEL/PATH
> cd ..
```


