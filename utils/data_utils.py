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
""" Multiple choice fine-tuning: utilities to work with multiple choice tasks of reading comprehension """

import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import List, Optional

import dgl
import numpy as np
import torch
import tqdm
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, RobertaTokenizer

import Config
from utils.data_utils_preprocess import construct_logic_graph, save_new_not_sentence_map

logger = logging.getLogger(__name__)


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"
    dev_and_test = 'dev_test'


class MyRobertaTokenizer(RobertaTokenizer):
    def build_inputs_with_special_tokens(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.eos_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        eos = [self.eos_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + eos

    def get_special_tokens_mask(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None,
            already_has_special_tokens: bool = False
    ) -> List[int]:
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formatted with special tokens for the model."
                )
            return list(map(lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0, token_ids_0))

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
            self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        eos = [self.eos_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + eos) * [0]
        return len(cls + token_ids_0 + sep + token_ids_1 + eos) * [0]


class DataProcessor:
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


@dataclass(frozen=True)
class InputExampleBertLogiGraph:
    example_id: str
    question: str
    context_origin: str
    endings_origin: List[str]
    context: List[List[str]]
    endings: List[List[str]]
    graphs: List[dgl.graph]
    edge_types: List[List[int]]
    edge_norms: List[List[int]]
    graph_node_nums: List[int]
    label: Optional[str]
    nodes_num: List[List[int]]
    base_nodes_ids: Optional[List[List[int]]]
    exten_nodes_ids: Optional[List[List[List[int]]]]
    exten_edges_ids: Optional[List[List[List[int]]]]


@dataclass(frozen=True)
class InputFeaturesBertLogiGraph:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    example_id: str
    input_ids_origin: List[List[int]]
    attention_mask_origin: Optional[List[List[int]]]
    token_type_ids_origin: Optional[List[List[int]]]
    input_ids: Optional[List[List[int]]]
    attention_mask: Optional[List[List[int]]]
    token_type_ids: Optional[List[List[int]]]
    graphs: List[List[torch.tensor]]
    graph_node_nums: List[int]
    edge_types: List[List[int]]
    edge_norms: List[List[float]]
    label: Optional[int]

    question_interval: Optional[List[List[int]]]
    node_intervals: Optional[List[List[List[int]]]]
    node_intervals_len: Optional[List[int]]
    context_interval: Optional[List[List[int]]]
    answer_interval: Optional[List[List[int]]]

    nodes_num: List[List[int]]

    base_nodes_ids: Optional[List[List[int]]]
    exten_nodes_ids: Optional[List[List[List[int]]]]
    exten_edges_ids: Optional[List[List[List[int]]]]

    @staticmethod
    def get_split_intervals(input_ids: List[List[int]], node_interval_padding_len=Config.node_interval_padding_len):
        node_sep_id = Config.tokenizer.convert_tokens_to_ids(Config.NODE_SEP_TOKEN)
        sep_id = Config.tokenizer.convert_tokens_to_ids(Config.SEP_TOKEN)
        seq_locs = [np.where((np.array(input_id) == sep_id))[0].tolist() for input_id in input_ids]
        assert sum(list(map(len, seq_locs))) == 2 * len(input_ids) and len(seq_locs) == len(input_ids)
        node_intervals = []
        node_sep_locs = [sorted([0] + np.where((np.array(input_id) == node_sep_id))[0].tolist() + seq_locs[index]) for
                         index, input_id in enumerate(input_ids)]
        node_intervals_num = []
        for index, node_sep_loc in enumerate(node_sep_locs):
            node_interval = []
            for i in range(len(node_sep_loc) - 1):
                node_interval.append([node_sep_loc[i] + 1, node_sep_loc[i + 1]])
            node_interval = list(filter(lambda x: x[1] - x[0] > 2, node_interval))
            node_intervals_num.append(len(node_interval))

            assert len(node_interval) < node_interval_padding_len
            while len(node_interval) < node_interval_padding_len:
                node_interval.append([Config.node_intervals_padding_id, Config.node_intervals_padding_id])
            node_intervals.append(node_interval)
        return None, node_intervals, node_intervals_num, None, None


class ProcessorBertLogiGraph(DataProcessor):
    """Processor for the RACE data set."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        return self._create_examples(f'{data_dir}/train.json', "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        return self._create_examples(f'{data_dir}/val.json', "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} test".format(data_dir))
        return self._create_examples(f'{data_dir}/test.json', "test")

    def get_labels(self):
        """See base class."""
        return list(range(4))

    def _read_json(self, input_dir):
        return json.load(open(input_dir, 'r', encoding='utf-8'))

    def _create_examples(self, data_dir, type):
        """Creates examples for the training and dev sets."""
        max_edge_num = 0
        max_node_num = 0
        datas = self._read_json(data_dir)
        examples = []
        for d in tqdm.tqdm(datas, total=len(datas), desc=f'preparing {type} data...'):
            label = d['label'] if 'label' in d else 0
            id_string = d['id_string']
            question = d['question']

            context, endings, graphs, node_sentences_a, node_sentences_b, relations, edge_norms, base_node_ids, cont_exten_node_ids, trans_exten_edge_ids = construct_logic_graph(
                id_string, return_base_nodes=True)
            exten_node_ids = None
            if base_node_ids is not None:
                for index in range(len(base_node_ids)):
                    assert len(base_node_ids[index]) <= Config.node_interval_padding_len
                    base_node_ids[index] += [Config.extension_padding_value] * (
                            Config.node_interval_padding_len - len(base_node_ids[index]))

            if cont_exten_node_ids is not None:
                exten_node_ids = cont_exten_node_ids
                for index in range(len(exten_node_ids)):
                    for inner_index in range(len(exten_node_ids[index])):
                        assert max(exten_node_ids[index][inner_index]) < len(node_sentences_a[index]) + len(
                            node_sentences_b[index]), f'id: {id_string}'
                        exten_node_ids[index][inner_index] += [Config.extension_padding_value] * (
                                4 - len(exten_node_ids[index][inner_index]))
                    assert len(exten_node_ids[
                                   index]) <= Config.extension_padding_len, f'exten node len: {len(exten_node_ids[index])}'
                    exten_node_ids[index] += [[Config.extension_padding_value] * 4] * (
                            Config.extension_padding_len - len(exten_node_ids[index]))

                exten_edge_ids = trans_exten_edge_ids
                for i in range(4):
                    for j in range(len(exten_edge_ids[i])):
                        assert len(exten_edge_ids[i][j]) == 3
                for index in range(len(exten_edge_ids)):
                    if os.path.exists('print_extension_len'):
                        print(len(exten_node_ids[index]))
                    assert len(exten_node_ids[
                                   index]) <= Config.extension_padding_len, f'exten edge len: {len(exten_edge_ids[index])}'
                    exten_edge_ids[index] += [[Config.extension_padding_value] * 3] * (
                            Config.extension_padding_len - len(exten_edge_ids[index]))

            max_edge_num = max(max_edge_num, max(list(map(len, relations))))
            max_node_num = max(max_node_num, max(list(map(lambda x: x.num_nodes(), graphs))))
            assert len(endings) == len(relations) == len(edge_norms)
            example = InputExampleBertLogiGraph(
                example_id=id_string,
                question=question,
                context_origin=context,
                endings_origin=endings,
                context=node_sentences_a,
                endings=node_sentences_b,
                graphs=graphs,
                edge_types=relations,
                edge_norms=edge_norms,
                graph_node_nums=[graph.num_nodes() for graph in graphs],
                label=label,
                nodes_num=[[len(n_a), len(n_b)] for n_a, n_b in zip(node_sentences_a, node_sentences_b)],
                base_nodes_ids=base_node_ids,
                exten_nodes_ids=exten_node_ids,
                exten_edges_ids=exten_edge_ids,
            )
            examples.append(example)

        logger.info(f'max edge num: {max_edge_num}')
        Config.max_edge_num = max(max_edge_num, Config.max_edge_num)
        Config.node_interval_padding_len = max(max_node_num, Config.node_interval_padding_len)
        return examples


class DatasetBertLogiGraph(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """

    features: List[InputFeaturesBertLogiGraph]

    def __init__(
            self,
            data_dir: str,
            tokenizer: PreTrainedTokenizer,
            task: str,
            max_seq_length: Optional[int] = None,
            overwrite_cache=False,
            mode: Split = Split.train,
    ):
        processor = processors[task]()

        cached_features_file = os.path.join(
            data_dir,
            "cached_{}_{}_{}_{}".format(
                mode.value,
                tokenizer.__class__.__name__,
                str(max_seq_length),
                os.environ['RUN_NAME'].split('@')[0],
            ),
        )

        logger.info(f'looking for cached file {cached_features_file}')

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        # lock_path = cached_features_file + ".lock"
        # with FileLock(lock_path):

        if os.path.exists(cached_features_file) and not overwrite_cache:
            logger.info(f"Loading features from cached file {cached_features_file}")
            self.features = torch.load(cached_features_file)
        else:
            logger.info(f"Creating features from dataset file at {data_dir}")
            label_list = processor.get_labels()
            if mode == Split.dev:
                examples = processor.get_dev_examples(data_dir)
            elif mode == Split.test:
                examples = processor.get_test_examples(data_dir)
            elif mode == Split.train:
                examples = processor.get_train_examples(data_dir)
            elif mode == Split.dev_and_test:
                examples = processor.get_dev_examples(data_dir) + processor.get_test_examples(data_dir)
            else:
                raise NotImplementedError
            logger.info("Training examples: %s", len(examples))
            self.features = convert_examples_to_features_graph_with_origin_rgcn(
                examples,
                label_list,
                max_seq_length,
                tokenizer, mode=Split.train,
            )
            save_new_not_sentence_map()
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(self.features, cached_features_file)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeaturesBertLogiGraph:
        return self.features[i]


def convert_examples_to_features_graph_with_origin_rgcn(
        examples: List[InputExampleBertLogiGraph],
        label_list: List[str],
        max_length: int,
        tokenizer: PreTrainedTokenizer, mode=Split.train
) -> List[InputFeaturesBertLogiGraph]:
    """
    Loads a data file into a list of `InputFeatures`
    """

    label_map = {label: i for i, label in enumerate(label_list)}

    logger.info(f'label map: {label_map}')
    trun_count = 0
    total_count = 1
    features = []
    t = tqdm.tqdm(enumerate(examples), desc=f"convert examples to features")
    for (ex_index, example) in t:
        if ex_index % 2000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        tokenizer_encode_method = tokenizer_encode

        attention_mask, attention_mask_origin, input_ids, input_ids_origin, label, token_type_ids, token_type_ids_origin, _trun_count, _total_count = tokenizer_encode_method(
            example, label_map, max_length, tokenizer)

        trun_count += _trun_count
        total_count += _total_count
        t.set_description(
            f'convert examples to features, trun count: {trun_count}, total_count: {total_count}, trun ratio: {trun_count / total_count}')

        edges = [[graph.edges()[0].numpy().tolist(), graph.edges()[1].numpy().tolist()] for graph in example.graphs]

        edge_types = example.edge_types
        edge_norms = example.edge_norms

        for i in range(len(edges)):
            assert len(edges[i][0]) == len(edges[i][1]) == len(edge_types[i]) == len(edge_norms[i])
            edges[i][0] = edges[i][0] + [-1] * (Config.max_edge_num - len(edges[i][0]))
            edges[i][1] = edges[i][1] + [-1] * (Config.max_edge_num - len(edges[i][1]))
            edge_types[i] = edge_types[i] + [-1] * (Config.max_edge_num - len(edge_types[i]))
            edge_norms[i] = edge_norms[i] + [-1] * (Config.max_edge_num - len(edge_norms[i]))

        get_split_intervals_method = InputFeaturesBertLogiGraph.get_split_intervals
        question_interval, node_intervals, node_intervals_num, context_interval, answer_interval = get_split_intervals_method(
            input_ids)
        new_feature = InputFeaturesBertLogiGraph(
            example_id=example.example_id,
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            graphs=edges,
            graph_node_nums=example.graph_node_nums,
            label=label,
            input_ids_origin=input_ids_origin,
            attention_mask_origin=attention_mask_origin,
            token_type_ids_origin=token_type_ids_origin,
            edge_types=example.edge_types,
            edge_norms=example.edge_norms,
            question_interval=question_interval,
            node_intervals=node_intervals,
            node_intervals_len=node_intervals_num,
            nodes_num=example.nodes_num,
            context_interval=context_interval,
            answer_interval=answer_interval,
            base_nodes_ids=example.base_nodes_ids,
            exten_nodes_ids=example.exten_nodes_ids,
            exten_edges_ids=example.exten_edges_ids
        )
        features.append(new_feature)

    for f in features[:2]:
        logger.info("*** Example ***")
        logger.info("feature: %s" % f)

    return features


def tokenizer_encode(example, label_map, max_length, tokenizer):
    choices_inputs = []
    choices_inputs_origin = []
    truncated_count = 0
    total_count = 0
    for index, context in enumerate(example.context):
        context_origin = example.context_origin
        ending_origin = example.endings_origin[index]

        assert isinstance(context_origin, str)
        assert isinstance(ending_origin, str)
        assert isinstance(context, list)
        assert isinstance(example.endings[index], list)

        inputs_origin = tokenizer(
            context_origin,
            example.question + Config.SEP_TOKEN + ending_origin,
            add_special_tokens=True,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_overflowing_tokens=True,
        )

        # [CLS] node_a_1 [N_SEP] node_a_2 [N_SEP] ... [N_SEP] node_a_n [SEP] node_b_1 [N_SEP] ... [N_SEP] node_b_n [SEP]
        text_a = Config.NODE_SEP_TOKEN.join(context)

        text_b = Config.NODE_SEP_TOKEN.join(example.endings[index])

        inputs = tokenizer(
            text_a,
            text_b,
            add_special_tokens=True,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_overflowing_tokens=True,
        )
        if "num_truncated_tokens" in inputs and inputs["num_truncated_tokens"] > 0:
            logger.info(
                "Attention! you are cropping tokens (swag task is ok). "
                "If you are training ARC and RACE and you are poping question + options,"
                "you need to try to use a bigger max seq length!"
            )
            truncated_count += 1
        total_count += 1

        choices_inputs.append(inputs)
        choices_inputs_origin.append(inputs_origin)
    label = label_map[int(example.label)]
    input_ids = [x["input_ids"][0] if Config.model_type == 'Bert' else x["input_ids"] for x in choices_inputs]
    attention_mask = (
        [x["attention_mask"][0] if Config.model_type == 'Bert' else x["attention_mask"] for x in
         choices_inputs] if "attention_mask" in choices_inputs[0] else None
    )
    token_type_ids = (
        [x["token_type_ids"][0] for x in choices_inputs] if "token_type_ids" in choices_inputs[0] else None
    )
    input_ids_origin = [x["input_ids"][0] if Config.model_type == 'Bert' else x["input_ids"] for x in
                        choices_inputs_origin]
    attention_mask_origin = (
        [x["attention_mask"][0] if Config.model_type == 'Bert' else x["attention_mask"] for x in
         choices_inputs_origin] if "attention_mask" in choices_inputs_origin[
            0] else None)
    token_type_ids_origin = (
        [x["token_type_ids"][0] for x in choices_inputs_origin] if "token_type_ids" in choices_inputs_origin[
            0] else None)
    return attention_mask, attention_mask_origin, input_ids, input_ids_origin, label, token_type_ids, token_type_ids_origin, truncated_count, total_count


processors = {"LogiGraph": ProcessorBertLogiGraph}
