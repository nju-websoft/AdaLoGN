# AdaLoGN: Adaptive Logic Graph Network for Reasoning-Based Machine Reading Comprehension

Code of paper "AdaLoGN: Adaptive Logic Graph Network for Reasoning-Based Machine Reading Comprehension".

## Requirements
```
torch==1.7.1
dgl-cu101==0.6.1
stanza==1.2.3
transformers==4.5.0
nltk
scikit-learn
pylev
```

## Data Preprocessing
We use [Graphene](https://github.com/Lambda-3/Graphene) to extract EDUs. We put all the contexts and options line by line in a .txt file and follow the instructions of Graphene to get EDUs outputs. Or you can also use our preprocessed file under directory ReclorDataset/LogiDataset.
We also provide cached file of preprocessed datas on [Google Drive](https://drive.google.com/drive/folders/1sAiOuH5HCucrgq2YZh-wm8s_khp6Mldv?usp=sharing). Download and put them under directory ReclorDataset/LogiQADataset.

## Evaluation
Checkpoints can be accessed on [Google Drive](https://drive.google.com/drive/folders/1sAiOuH5HCucrgq2YZh-wm8s_khp6Mldv?usp=sharing).
```shell
export MODE=eval_only
bash scripts/LogiGraph_Roberta.sh /PATH/TO/RECLOR/CHECKPOINTS  ## ReClor evaluation
bash scripts/LogiGraph_Roberta_LogiQA.sh /PATH/TO/LOGIQA/CHECKPOINTS  ## LogiQA evaluation
```

For ReClor dataset, we submit prediction file on [ReClor Leaderboard](https://eval.ai/web/challenges/challenge-page/503/leaderboard/1347) and AdaLoGN achieves Rank #10 on leaderboard (03/15/2022).

## Training
You can also install [wandb](https://docs.wandb.ai/quickstart) and set ``export WANDB_DISABLED=false`` in training scripts to visualize the training process.
```shell
export MODE=do_train
bash scripts/LogiGraph_Roberta.sh /PATH/TO/ROBERTA/LARGE  ## ReClor evaluation
bash scripts/LogiGraph_Roberta_LogiQA.sh /PATH/TO/ROBERTA/LARGE  ## LogiQA evaluation
```

## Citation
To be updated.