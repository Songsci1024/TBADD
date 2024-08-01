# Risk of Text Backdoor Attacks under Dataset Distillation
Implementation of "Risk of Text Backdoor Attacks under Dataset Distillation" (accepted by ISC2024)

Abstract: Dataset distillation aims to transfer knowledge from large training datasets into smaller datasets to enable rapid training of neural networks while maintaining the original dataset's performance. However, current research on dataset distillation primarily focuses on balancing resource utilization and model capability, with limited discussion on the associated security risks, especially in the natural language processing (NLP) domain. In this paper, we focus on backdoor attacks on data distilled from text datasets. Specifically, we inject triggers into the synthetic dataset during the distillation process, rather than during the model training phase. We propose a framework for backdoor attacks in the context of text dataset distillation, termed Text Backdoor Attack under Dataset Distillation (TBADD). This framework is broadly applicable to backdoor attack methods based on dataset poisoning principles. It achieves an optimal balance between clean sample accuracy (CACC) and attack success rate(ASR) by separating clean and poisoned samples in the validation set and evaluating the distilled dataset's performance through weighted assessment. Experimental comparisons using four popular backdoor attacks on two text classification tasks demonstrate that TBADD can achieve attack success rates comparable to those of models trained with the original dataset without significantly compromising the original task performance. Under two visible backdoor attacks, the ASR approaches 100\%, while under two invisible backdoor attacks, the average ASR is 83\%.

## Introduction
The paper code is developed based on [OpenBackdoor](https://github.com/thunlp/OpenBackdoor) and [Dataset Distillation with Attention Labels](https://github.com/arumaekawa/dataset-distillation-with-attention-labels). Its primary objective is to investigate the threat of backdoor attacks in the text domain on dataset distillation, facilitating further in-depth research.

![framework](pics/framework.png)


## Prepare
### Configure Develop Environment:
```shell
conda create -n tbadd python=3.11.5
conda activate tbadd
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

```shell
# Install java under ubuntu environment used for SynBKD
sudo apt-get update
sudo apt-get install -y openjdk-8-jdk
java -version
```

### Download Datasets
```shell
cd datasets
bash download_sentiment_analysis.sh
cd ..
```

### Download Other Resources
You can download the organized resources from [the cloud storage](https://drive.google.com/drive/folders/1BQEMnyHTw6Xo4gxazXN4mFSDJx0M03ER?usp=sharing) or using the original link below, and store them in their respective locations.
1. [bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased): Move the downloaded folder to the root directory of TBADD.
2. [bible](https://huggingface.co/lievan/bible): Construct GPT-2 pretrained parameters for a Bible-style trigger([other styles](https://huggingface.co/lievan/)). Save them to the `./data/` directory.

## Run code
### Attack
```
python demo_dd_attack.py --config_path configs/badnets_config_dd.json

python demo_dd_attack.py --config_path configs/addsent_config_dd.json

python demo_dd_attack.py --config_path configs/syn_config_dd.json

python demo_dd_attack.py --config_path configs/style_config_dd.json
```

### Defense

Using trained target model parameters for defense (recommended):

1. Run `demo_dd_attack.py` to obtain the trained model parameters `trained_model.pth` (located in the `save/` directory) after training on the distilled dataset.
2. Specify the path of `trained_model.pth` in the `defender.pretrained_model_path_dir` field within the `config/dd_defend` directory. Like `"pretrained_model_path_dir": "save/dpc_1.step_1.bsz_1.label_hard.al_cls.lr_init_0.01.distill_lr_0.01/2024-07-02.12-56-31"`
3. Execute the demo_dd_defend.py file using scripts [RAP](#rap),[STRIP](#strip)

If not using model parameters, train on the distilled dataset and then proceed with defense: Simply run demo_dd_defend.py.
#### RAP
```
python demo_dd_defend.py --config_path configs/dd_defend/rap/badnets_config_dd.json

python demo_dd_defend.py --config_path configs/dd_defend/rap/addsent_config_dd.json

python demo_dd_defend.py --config_path configs/dd_defend/rap/syn_config_dd.json

python demo_dd_defend.py --config_path configs/dd_defend/rap/style_config_dd.json
```

#### STRIP

```
python demo_dd_defend.py --config_path configs/dd_defend/strip/badnets_config_dd.json

python demo_dd_defend.py --config_path configs/dd_defend/strip/addsent_config_dd.json

python demo_dd_defend.py --config_path configs/dd_defend/strip/syn_config_dd.json

python demo_dd_defend.py --config_path configs/dd_defend/strip/style_config_dd.json
```

## Check dataset distillation results with mlflow
```
mlflow server --backend-store-uri ./mlruns --host 0.0.0.0
open http://localhost:5000
```

## Results
**Attack result**

![attack result](pics/result%20attack.png)

**Defense result**

![attack defend](pics/result%20defend.png)
