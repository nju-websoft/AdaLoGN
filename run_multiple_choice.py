# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for multiple choice (Bert, Roberta, XLNet)."""
import json
import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional

import dgl
import numpy as np
import torch
import transformers
from sklearn.metrics import accuracy_score
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
    set_seed, Trainer,
)
from transformers.trainer_utils import is_main_process

import Config
from models import RobertaAdaLoGN
from utils.data_utils import DatasetBertLogiGraph, processors, MyRobertaTokenizer, Split

logger = logging.getLogger(__name__)

model_class = {"LogiGraph": RobertaAdaLoGN}

dataset_class = {"LogiGraph": DatasetBertLogiGraph}


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default='/home/xli/bert_model_en',
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    gnn_layers_num: int = field(default=-1)
    model_type: str = field(default='Bert')
    base_num: int = field(default=6)
    rgcn_relation_nums: int = field(default=6)
    dropout: float = field(default=0.1)
    results_output_dir: str = field(default='results')
    pooling_type: str = field(default='none')
    label_smoothing: bool = field(default=True)
    extension_threshold: float = field(default=0.6)
    label_smoothing_factor2: float = field(default=0.25)
    eval_only: bool = field(default=False)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    task_name: str = field(metadata={"help": "The name of the task to train on: " + ", ".join(processors.keys())})
    data_dir: str = field(metadata={"help": "Should contain the data files for the task."})
    max_seq_length: int = field(
        default=256,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    Config.rgcn_relation_nums = model_args.rgcn_relation_nums

    Config.model_args = model_args

    if (
            os.path.exists(training_args.output_dir)
            and os.listdir(training_args.output_dir)
            and training_args.do_train
            and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)
    dgl.seed(training_args.seed)
    Config.seed = training_args.seed
    Config.extension_threshold = model_args.extension_threshold

    try:
        processor = processors[data_args.task_name]()
        label_list = processor.get_labels()
        num_labels = len(label_list)
    except KeyError:
        raise ValueError("Task not found: %s" % (data_args.task_name))

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
    )

    tokenizer_class = MyRobertaTokenizer if model_args.model_type == 'Roberta' else AutoTokenizer

    logger.info(f'model type: {model_args.model_type}')
    logger.info(f'tokenizer class: {tokenizer_class.__name__}')

    tokenizer = tokenizer_class.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    tokenizer.add_special_tokens({'additional_special_tokens': [Config.NODE_SEP_TOKEN]})
    Config.tokenizer = tokenizer
    logger.info(f'roberta class: {model_class}')

    def model_init():
        return model_class[data_args.task_name].from_pretrained(
            model_args.model_name_or_path,
            from_tf=False,
            config=config,
            cache_dir=model_args.cache_dir, )

    logger.info(f'model type: {Config.model_type}')

    # Get datasets
    train_dataset = (
        dataset_class[data_args.task_name](
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            task=data_args.task_name,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.train,
        )
        if training_args.do_train
        else None
    )

    eval_dataset = (
        dataset_class[data_args.task_name](
            data_dir=data_args.data_dir,
            tokenizer=tokenizer,
            task=data_args.task_name,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.dev_and_test,
        )
        if training_args.do_eval
        else None
    )

    def compute_metrics(p: EvalPrediction) -> Dict:
        _preds = p.predictions
        print(f'pred: {_preds}')
        _preds[_preds != _preds] = 0

        preds = [x.tolist().index(max(x.tolist())) for x in p.predictions]
        torch.cuda.empty_cache()
        if Config.eval_test:
            dev_len = 500 if os.environ['DATASET_DIR'] == 'ReclorDataset' else (len(preds) // 2)
            dev_acc = accuracy_score(p.label_ids[:dev_len], preds[:dev_len])

            result = {"acc_dev": dev_acc, }

            if not os.environ['DATASET_DIR'] == 'ReclorDataset':
                result["acc_test"] = accuracy_score(p.label_ids[dev_len:], preds[dev_len:])
                _p = [[math.e ** xx for xx in x] for x in _preds[:len(preds) // 2]]
                dev_loss = sum([-math.log(x[p.label_ids[index]] / sum(x)) for index, x in enumerate(_p)]) / len(_p)
                result['dev_loss_dev'] = dev_loss
                _p = [[math.e ** xx for xx in x] for x in _preds[len(preds) // 2:]]
                dev_loss = sum([-math.log(x[p.label_ids[index]] / sum(x)) for index, x in enumerate(_p)]) / len(_p)
                result['dev_loss_test'] = dev_loss
            else:
                _p = [[math.e ** xx for xx in x] for x in _preds[:500]]
                dev_loss = sum([-math.log(x[p.label_ids[index]] / sum(x)) for index, x in enumerate(_p)]) / len(_p)
                result['dev_loss2'] = dev_loss
                assert sum(p.label_ids[dev_len:]) == 0, f'{p.label_ids[dev_len:]}'

            if not os.path.exists(os.path.join('results', training_args.run_name)):
                logger.info(f'mkdir {os.path.join("results", training_args.run_name)}')
                os.mkdir(os.path.join('results', training_args.run_name))

            count = 0
            save_path = f'test_{dev_acc}_{count}.npy'
            while os.path.exists(os.path.join('results', training_args.run_name, save_path)):
                count += 1
                save_path = f'test_{dev_acc}_{count}.npy'
            np.save(os.path.join('results', training_args.run_name, save_path), preds[dev_len:])

            return result
        else:
            return {"acc": accuracy_score(p.label_ids, preds)}

    trainer = Trainer(
        model_init=model_init,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=None,
    )

    # Training
    if training_args.do_train and not model_args.eval_only:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        result = trainer.evaluate()
        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key, value in result.items():
                logger.info("  %s = %s", key, value)
                writer.write("%s = %s\n" % (key, value))
            results.update(result)
    return results


if __name__ == "__main__":
    main()
