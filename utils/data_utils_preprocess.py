import copy
import json
import os
import random
import re
import sys
from typing import List, Tuple

import dgl
import networkx as nx
import numpy as np
import pylev
import stanza
from tqdm import tqdm

import Config
from utils.pos_neg_transfer import pos_neg_convert

dataset_dir = os.environ['DATASET_DIR']


def load_data():
    train_dataset = json.load(open(f'{dataset_dir}/train.json', 'r', encoding='utf-8'))
    train_dataset = dict([(data['id_string'], data) for data in train_dataset])

    val_dataset = json.load(open(f'{dataset_dir}/val.json', 'r', encoding='utf-8'))
    val_dataset = dict([(data['id_string'], data) for data in val_dataset])

    test_dataset = json.load(open(f'{dataset_dir}/test.json', 'r', encoding='utf-8'))
    test_dataset = dict([(data['id_string'], data) for data in test_dataset])

    return train_dataset, val_dataset, test_dataset


train_dataset, val_dataset, test_dataset = load_data()

dataset = {**train_dataset, **val_dataset, **test_dataset}


def load_all_subsentences_with_logic():
    subsentences_with_logic = {}
    for filename in os.listdir(f'{dataset_dir}/flat'):
        if not filename.endswith('.flat'):
            continue
        flat_datas = open(f'{dataset_dir}/flat/{filename}', 'r', encoding='utf-8').read().split('\n')
        flat_datas = list(filter(lambda x: x.strip() != '', flat_datas))
        new_flat_datas = []
        for data in flat_datas:
            if data not in new_flat_datas:
                new_flat_datas.append(data)
        flat_datas = new_flat_datas
        for line in flat_datas:
            line = line.split('\t')
            origin_sentence, hash_id, num, subsentence = line[:4]
            logic_relation = line[4:]
            logic_relation = list(map(lambda x: x.split('('), logic_relation))
            logic_relation = list(
                map(lambda x: (x[0].replace('L:', '').replace('S:', ''), x[1].replace(')', '')), logic_relation))
            origin_sentence = origin_sentence.replace('baseDir', '')
            if ord('a') <= ord(origin_sentence[-1]) <= ord('z') or ord('A') <= ord(origin_sentence[-1]) <= ord('Z'):
                origin_sentence += '.'
            origin_sentence = origin_sentence.replace('ttherefore', 'therefore').replace('Ttherefore', 'Therefore')
            origin_sentence = origin_sentence.replace('<b>', '').replace('baseDir', '').replace('</b>', '').replace(
                '<i>', '').replace('</i>',
                                   '').replace(
                '<a>', '').replace('</a>', '').replace('baseDir', '').strip()
            if origin_sentence not in subsentences_with_logic:
                subsentences_with_logic[origin_sentence] = {}
            subsentences_with_logic[origin_sentence][hash_id] = {'subsentence': subsentence,
                                                                 'logic_relation': logic_relation}
    return subsentences_with_logic


subsentences_with_logic = load_all_subsentences_with_logic()


def _remove_last_dot(s: str) -> str:
    if s.endswith('.'):
        s = s[:-1].strip()
    return s


def _lower_first_letter(s: str) -> str:
    s = s.strip()
    if ord('A') <= ord(s[0]) <= ord('Z'):
        s = s[0].lower() + s[1:]
    return s


def _subsentence_format(s: str):
    for _ in range(5):
        s = s.strip()
        s = s.replace(' .', '.').replace(' ,', '').replace(' !', '!').replace(' ?', '?').replace(' :', ':')
        s = s.replace(' \' s ', '\'s ')
        s = s.replace(' \'s ', '\'s ')
        s = s.replace('  ', ' ')
    if s[-1] not in '.?;':
        s = s + '.'
    return s


EDUs = json.load(open(f'{dataset_dir}/{dataset_dir}_EDUs.json', 'r', encoding='utf-8'))
EDUs_tmp = {}
for sentence in EDUs:
    EDUs_tmp[sentence.strip()] = EDUs[sentence]
EDUs = EDUs_tmp


def get_element_map(text):
    assert text in EDUs, text
    element_map = {}
    for sentence in EDUs[text]['sentences']:
        element_map = {**element_map, **sentence['elementMap']}
    del_hash_ids = []
    for hash_id in element_map:
        if len(element_map[hash_id]['text'].split(' ')) < 3 and len(element_map) > 1:
            del_hash_ids.append(hash_id)
    for hash_id in del_hash_ids:
        del element_map[hash_id]
    element_map_str = json.dumps(element_map)
    for hash_id in del_hash_ids:
        element_map_str = element_map_str.replace(hash_id, 'deleted')
    return edu_merge(json.loads(element_map_str))


def sentence_format(s):
    s = s.replace(' \' s', '\'s').replace(' .', '.')
    return s


def edu_merge(edus):
    new_edus = copy.deepcopy(edus)
    processed_texts = []
    del_hash_ids_map = {}
    for hash_id in edus:
        text = edus[hash_id]['text']
        if text in processed_texts:
            continue
        linked_contexts = []
        del_hash_ids = []
        for hash_id2 in edus:
            if edus[hash_id2]['text'] == text:
                linked_contexts += edus[hash_id2]['linkedContexts']
                del new_edus[hash_id2]
                del_hash_ids.append(hash_id2)
        element_map = edus[hash_id]
        element_map['linkedContexts'] = linked_contexts
        new_edus[hash_id] = element_map
        processed_texts.append(text)
        del_hash_ids_map[hash_id] = del_hash_ids
    edus_str = json.dumps(new_edus)
    for hash_id in del_hash_ids_map:
        for old_hash_id in del_hash_ids_map[hash_id]:
            edus_str = edus_str.replace(old_hash_id, hash_id)
    return json.loads(edus_str)


def construct_relation_graph_new(id: str):
    context, answers = dataset[id]['context'], dataset[id]['answers']
    graphs = []
    relations = []
    context = context.replace('<b>', '').replace('baseDir', '').replace('</b>', '').replace('<i>', '').replace('</i>',
                                                                                                               '').replace(
        'Ttherefore', 'Therefore').replace('ttherefore', 'therefore').strip()
    relation_types = ['AND', 'OR', 'IMP', 'IMP_REV', 'OTHER', 'NOT']
    node_sentences_a, node_sentences_b = [], []
    context_edus = get_element_map(context)

    # context, answers, graphs, node_sentences_a, node_sentences_b, relations

    for __index, sentence in enumerate(answers):

        sentences_a, sentences_b = [], []
        main_chains_sentences = []
        raw_relations = {}

        def add_nodes(_text_hash_id, a_or_b):
            if _text_hash_id in sentences_a and a_or_b == 'a':
                return
            if _text_hash_id in sentences_b and a_or_b == 'b':
                return
            if a_or_b == 'a':
                sentences_a.append(_text_hash_id)
            elif a_or_b == 'b':
                sentences_b.append(_text_hash_id)
            else:
                raise NotImplementedError

        def add_edges(_text_hash_id_src, _text_hash_id_dst, relation_id):
            if f'{_text_hash_id_src}_{_text_hash_id_dst}' not in raw_relations:
                raw_relations[f'{_text_hash_id_src}_{_text_hash_id_dst}'] = relation_id

        answer_sentence = sentence.replace('<b>', '').replace('</b>', '').replace('<i>', '').replace('</i>',
                                                                                                     '').replace(
            'Ttherefore', 'Therefore').replace('ttherefore', 'therefore').replace('baseDir', '').strip()

        answer_edus = get_element_map(answer_sentence)
        edus = {**context_edus, **answer_edus}

        for hash_id in edus:
            main_chains_sentences.append(hash_id)
        sentences_a = list(context_edus.keys())
        sentences_b = list(answer_edus.keys())

        for hash_id in edus:
            for linked_edu in edus[hash_id]['linkedContexts']:
                target_id = linked_edu['targetID']
                relation = linked_edu['relation']

                if target_id == 'deleted':
                    continue
                if relation in ['UNKNOWN_SUBORDINATION', 'ATTRIBUTION', 'SPATIAL']:
                    continue
                elif relation in ['BACKGROUND', 'CAUSE', 'CONDITION', 'PURPOSE', 'CAUSE_C']:
                    add_nodes(target_id,
                              'a' if target_id in context_edus else ('b' if target_id in answer_edus else 'error'))
                    add_edges(hash_id, target_id, relation_types.index('IMP'))
                    add_edges(target_id, hash_id, relation_types.index('IMP_REV'))
                elif relation in ['RESULT', 'RESULT_C']:
                    add_nodes(target_id,
                              'a' if target_id in context_edus else ('b' if target_id in answer_edus else 'error'))
                    add_edges(target_id, hash_id, relation_types.index('IMP'))
                    add_edges(hash_id, target_id, relation_types.index('IMP_REV'))
                elif relation in ['LIST', 'CONTRAST']:
                    add_nodes(target_id,
                              'a' if target_id in context_edus else ('b' if target_id in answer_edus else 'error'))
                    add_edges(target_id, hash_id, relation_types.index('AND'))
                    add_edges(hash_id, target_id, relation_types.index('AND'))
                elif relation in ['DISJUNCTION']:
                    add_nodes(target_id,
                              'a' if target_id in context_edus else ('b' if target_id in answer_edus else 'error'))
                    add_edges(target_id, hash_id, relation_types.index('OR'))
                    add_edges(hash_id, target_id, relation_types.index('OR'))
                elif relation in ['IDENTIFYING_DEFINITION', 'ELABORATION', 'DESCRIBING_DEFINITION',
                                  'TEMPORAL_BEFORE_C', 'TEMPORAL_AFTER_C', 'TEMPORAL_BEFORE',
                                  'TEMPORAL_AFTER']:
                    continue
                elif relation in ['NOUN_BASED']:
                    continue
                else:
                    raise NotImplementedError

        for main_chain_id in range(len(main_chains_sentences) - 1):
            add_edges(main_chains_sentences[main_chain_id], main_chains_sentences[main_chain_id + 1],
                      relation_types.index('OTHER'))
            add_edges(main_chains_sentences[main_chain_id + 1], main_chains_sentences[main_chain_id],
                      relation_types.index('OTHER'))
        all_sentences = sentences_a + sentences_b
        edges = [(all_sentences.index(id.split('_')[0]), all_sentences.index(id.split('_')[1])) for id in raw_relations]
        edges = ([s[0] for s in edges], [s[1] for s in edges])
        relations.append([raw_relations[id] for id in raw_relations])
        sentences_a = [sentence_format(edus[hash_id]['text']) for hash_id in sentences_a]
        sentences_b = [sentence_format(edus[hash_id]['text']) for hash_id in sentences_b]
        node_sentences_a.append(sentences_a)
        node_sentences_b.append(sentences_b)
        all_sentences = sentences_a + sentences_b

        nodes_num = len(all_sentences)
        edges_a, edges_b = edges[0], edges[1]
        for index1 in range(nodes_num):
            sentence_index1 = all_sentences[index1]
            for index2 in range(index1 + 1, nodes_num, 1):
                sentence_index2 = all_sentences[index2]
                if pylev.levenschtein(sentence_index1.split(' '), sentence_index2.split(' ')) > 2:
                    continue
                if sentence_index1 in get_from_new_not_sentence_map(
                        sentence_index2) or sentence_index2 in get_from_new_not_sentence_map(sentence_index1):
                    if [index1, index2] in [[a, b] for a, b in zip(edges_a, edges_b)]:
                        continue
                    edges_a.append(index1)
                    edges_b.append(index2)
                    edges_a.append(index2)
                    edges_b.append(index1)
                    relations[__index].append(relation_types.index('NOT'))

        graphs.append(dgl.graph((edges_a, edges_b)))
        assert is_dgl_graph_connected(graphs[__index])

    return context, answers, graphs, node_sentences_a, node_sentences_b, relations


def get_edge_norm(edge_relations, graph, graph_edges):
    edge_norm = []
    for i in range(graph.num_edges()):
        in_node_id = graph_edges[1][i]
        current_edge_type = edge_relations[i]
        current_edge_type_count = 0
        for j in range(graph.num_edges()):
            if graph_edges[1][j] != in_node_id:
                continue
            if edge_relations[j] == current_edge_type:
                current_edge_type_count += 1
        edge_norm.append(1.0 / current_edge_type_count)
    return edge_norm


def merge(graph: dgl.graph, relations, merge_relation_type, node_sentence_a_len, forced_merge=False,
          base_node_ids=None):
    edges = [graph.edges()[0].numpy().tolist(), graph.edges()[1].numpy().tolist()]

    merge_nodes = []
    for node_id in range(graph.num_nodes()):
        if node_id in merge_nodes or node_id - 1 in merge_nodes:
            continue
        if node_id == node_sentence_a_len - 1:
            continue
        if not forced_merge:
            next_node_id = node_id + 1
        else:
            loop_count = 0
            node_id = -1
            while node_id == -1 or node_id in merge_nodes or node_id + 1 in merge_nodes or node_id - 1 in merge_nodes:
                node_id = random.randint(0, graph.num_nodes() - 2)
                if node_id == node_sentence_a_len - 1:
                    node_id = -1
                    continue
                loop_count += 1
                if loop_count > graph.num_nodes() // 2:
                    break
            if node_id == -1 or node_id == node_sentence_a_len - 1:
                continue
            next_node_id = node_id + 1
        if node_id not in base_node_ids or node_id + 1 not in base_node_ids:
            continue

        is_other_edge_type = False
        is_connected = False
        for i in range(len(edges[0])):
            if edges[0][i] == node_id or edges[1][i] == node_id or edges[0][i] == next_node_id or edges[1][
                i] == next_node_id:
                if relations[i] != merge_relation_type and not forced_merge:
                    is_other_edge_type = True
                    break
            if edges[0][i] == node_id and edges[1][i] == next_node_id or edges[0][i] == next_node_id and edges[1][
                i] == node_id:
                is_connected = True
        if is_other_edge_type or not is_connected:
            continue
        assert node_id != node_sentence_a_len - 1
        merge_nodes.append(node_id)

        for i in range(len(edges[0])):
            if edges[0][i] == next_node_id:
                edges[0][i] = node_id
            if edges[1][i] == next_node_id:
                edges[1][i] = node_id


    return merge_nodes, edges


def random_delete_edges(context, answers, graphs, _node_sentences_a, _node_sentences_b, relations, selected_i,
                        min_edge_nums=Config.truncate_edges_num):
    for i in range(4):
        if selected_i is not None and selected_i != i:
            continue
        assert len(_node_sentences_a[i]) + len(_node_sentences_b[i]) == graphs[i].num_nodes()
        loop_count = 0
        while graphs[i].num_edges() > min_edge_nums and loop_count < 60:
            loop_count += 1
            del_edge_id = random.randint(0, graphs[i].num_edges() - 1)
            edges_a, edges_b = graphs[i].edges()[0].numpy().tolist(), graphs[i].edges()[1].numpy().tolist()

            edges_a_tmp = edges_a[:del_edge_id] + edges_a[del_edge_id + 1:]
            edges_b_tmp = edges_b[:del_edge_id] + edges_b[del_edge_id + 1:]
            del_node_a, del_node_b = edges_a[del_edge_id], edges_b[del_edge_id]

            new_graph = dgl.graph((edges_a_tmp, edges_b_tmp))
            if not is_dgl_graph_connected(new_graph, show_graph=False) or not new_graph.num_nodes() == graphs[
                i].num_nodes():
                continue

            graphs[i] = new_graph
            del relations[i][del_edge_id]
        assert len(_node_sentences_a[i]) + len(_node_sentences_b[i]) == graphs[i].num_nodes()

    return context, answers, graphs, _node_sentences_a, _node_sentences_b, relations, None


def construct_relation_graph_merger_nodes(id, show_graph=False, graph_type=4, context=None, answers=None,
                                          graphs=None, _node_sentences_a=None, _node_sentences_b=None, relations=None,
                                          selected_i=None, merge_4=False, return_merge_nodes=False, forced_merge=False,
                                          base_node_ids=None):
    if context is None:
        assert 1 <= graph_type <= 5
        assert id is not None
        context, answers, graphs, _node_sentences_a, _node_sentences_b, relations, _ = getattr(sys.modules[__name__],
                                                                                               f'construct_relation_graph{graph_type}')(
            id=id, show_graph=show_graph)

    ret_merge_nodes = []
    for i in range(4):
        if selected_i is not None and selected_i != i:
            continue
        node_sentences_a = _node_sentences_a[i]
        node_sentences_b = _node_sentences_b[i]

        merge_nodes, edges = merge(graphs[i], relations[i], merge_relation_type=4,
                                   node_sentence_a_len=len(node_sentences_a), forced_merge=forced_merge,
                                   base_node_ids=base_node_ids[i])

        ret_merge_nodes.append(merge_nodes)
        for node in merge_nodes:
            if node < len(node_sentences_a):
                node_sentences_a[node] = node_sentences_a[node] + ' ' + node_sentences_a[node + 1]
                node_sentences_a[node + 1] = 'NULL'
            else:
                _node = node - len(node_sentences_a)
                node_sentences_b[_node] = node_sentences_b[_node] + ' ' + node_sentences_b[_node + 1]
                node_sentences_b[_node + 1] = 'NULL'

        new_node_num = graphs[i].num_nodes() - len(merge_nodes)

        for index in range(new_node_num):
            while (index not in edges[0] + edges[1]) and len(edges[0] + edges[1]) > 0 and index <= max(
                    edges[0] + edges[1]):
                for edge_index in range(len(edges[0])):
                    if edges[0][edge_index] > index:
                        edges[0][edge_index] -= 1
                    if edges[1][edge_index] > index:
                        edges[1][edge_index] -= 1

        node_sentences_a = list(filter(lambda x: x != 'NULL', node_sentences_a))
        node_sentences_b = list(filter(lambda x: x != 'NULL', node_sentences_b))

        for index1 in range(len(edges[0])):
            if edges[0][index1] == -1:
                continue
            for index2 in range(index1 + 1, len(edges[0])):
                if edges[0][index1] == edges[0][index2] and edges[1][index1] == edges[1][index2]:
                    edges[0][index2] = edges[1][index2] = -1
                    relations[i][index2] = -1

        edges = (list(filter(lambda x: x != -1, edges[0])), list(filter(lambda x: x != -1, edges[1])))
        graphs[i] = dgl.graph(edges)
        relations[i] = list(filter(lambda x: x != -1, relations[i]))

        _node_sentences_a[i] = node_sentences_a
        _node_sentences_b[i] = node_sentences_b

        assert len(node_sentences_a) != 0
        assert len(node_sentences_b) != 0

    if return_merge_nodes:
        return context, answers, graphs, _node_sentences_a, _node_sentences_b, relations, [
            get_edge_norm(r, g, g.edges()) for r, g in zip(relations, graphs)], ret_merge_nodes
    return context, answers, graphs, _node_sentences_a, _node_sentences_b, relations, [get_edge_norm(r, g, g.edges())
                                                                                       for r, g in
                                                                                       zip(relations, graphs)]


def move_node_sentences(edges, node_sentences_a, node_sentences_b, node_sentences_c, node_sentences_d,
                        base_node_id=None, cont_exten_node_id=None, trans_exten_edge_id=None):
    new_sentences_a = node_sentences_a + node_sentences_c
    new_sentences_b = node_sentences_b + node_sentences_d
    new_edges = ([], [])
    old_sentences = node_sentences_a + node_sentences_b + node_sentences_c + node_sentences_d
    new_sentences = new_sentences_a + new_sentences_b
    if base_node_id is not None:
        new_base_node_id = [new_sentences.index(old_sentences[id]) for id in base_node_id]
        new_cont_exten_node_id = [[new_sentences.index(old_sentences[id]) for id in n] for n in cont_exten_node_id]
        new_trans_exten_edge_id = [[new_sentences.index(old_sentences[id]) for id in n[:2]] + [n[2]] for n in
                                   trans_exten_edge_id]

    for i in range(len(edges[0])):
        s_a = old_sentences[edges[0][i]]
        s_b = old_sentences[edges[1][i]]

        new_edges[0].append((new_sentences_a + new_sentences_b).index(s_a))
        new_edges[1].append((new_sentences_a + new_sentences_b).index(s_b))
    if base_node_id is not None:
        return new_edges, new_sentences_a, new_sentences_b, new_base_node_id, new_cont_exten_node_id, new_trans_exten_edge_id
    return new_edges, new_sentences_a, new_sentences_b


nlp = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,sentiment,lemma,depparse', use_gpu=False)
new_not_sentence_map = json.load(open('datas/negative_sentences_map.json', 'r', encoding='utf-8'))


def get_from_new_not_sentence_map(s):
    if s in new_not_sentence_map:
        return new_not_sentence_map[s]
    ret = pos_neg_convert(s, nlp, return_sentiment=False, return_adv_neg=True)
    new_not_sentence_map[s] = ret
    return ret


def save_new_not_sentence_map():
    output = open('datas/negative_sentences_map.json', 'w', encoding='utf-8')
    json.dump(new_not_sentence_map, output, ensure_ascii=False)
    output.close()


def construct_relation_graph_extension(id: str, context=None, graphs=None, node_sentences_a=None,
                                       node_sentences_b=None, relations=None,
                                       return_base_nodes=False):
    assert context is not None

    base_node_ids = [list(range(len(node_sentences_a[index]) + len(node_sentences_b[index]))) for index in range(4)]

    context, answers = dataset[id]['context'], dataset[id]['answers']
    context = context.replace('<b>', '').replace('baseDir', '').replace('</b>', '').replace('<i>', '').replace('</i>',
                                                                                                               '').replace(
        'baseDir', '').strip()
    answers = [
        a.replace('<b>', '').replace('baseDir', '').replace('</b>', '').replace('<i>', '').replace('</i>', '').strip()
        for a in answers]
    node_sentences_c = [[], [], [], []]
    node_sentences_d = [[], [], [], []]
    relation_types = ['AND', 'OR', 'IMP', 'IMP_REV', 'OTHER', 'NOT']

    cont_exten_node_ids = [[], [], [], []]
    trans_exten_edge_ids = [[], [], [], []]

    def add_edge(_a, _b, _edge_type, i):
        assert _edge_type in relation_types
        assert 0 <= _a <= len(node_sentences_a[i]) + len(node_sentences_b[i]) + len(node_sentences_c[i]) + len(
            node_sentences_d[i])
        assert 0 <= _b <= len(node_sentences_a[i]) + len(node_sentences_b[i]) + len(node_sentences_c[i]) + len(
            node_sentences_d[i])
        graphs[i].add_edge(_a, _b)
        relations[i].append(relation_types.index(_edge_type))

    def add_node(_node_sentence, i, sentence_type):
        if _node_sentence in node_sentences_a[i] or _node_sentence in node_sentences_b[i] or _node_sentence in \
                node_sentences_c[i] or _node_sentence in node_sentences_d[i]:
            return (node_sentences_a[i] + node_sentences_b[i] + node_sentences_c[i] + node_sentences_d[i]).index(
                _node_sentence)
        assert sentence_type != 'EXIST'
        if sentence_type == 'c':
            node_sentences_c[i].append(_node_sentence)
        elif sentence_type == 'd':
            node_sentences_d[i].append(_node_sentence)
        else:
            raise NotImplementedError
        return len(node_sentences_a[i]) + len(node_sentences_b[i]) + len(node_sentences_c[i]) + len(
            node_sentences_d[i]) - 1

    imp_relation_id = relation_types.index('IMP')
    or_relation_id = relation_types.index('OR')
    and_relation_id = relation_types.index('AND')
    other_relation_id = relation_types.index('OTHER')

    for i in range(4):
        tmp_edges_id = []
        node_sentences_a[i] = [s + '_c' for s in node_sentences_a[i]]
        node_sentences_b[i] = [s + '_d' for s in node_sentences_b[i]]

        max_extension_depth = 2

        for relation_index in range(len(relations[i])):
            if len(trans_exten_edge_ids[i]) > Config.extension_padding_len:
                break
            if relations[i][relation_index] == other_relation_id:
                edge_a, edge_b = graphs[i].edges()[0].numpy().tolist()[relation_index], \
                                 graphs[i].edges()[1].numpy().tolist()[relation_index]
                for edge_index in range(len(graphs[i].edges()[0].numpy().tolist())):
                    if graphs[i].edges()[0].numpy().tolist()[edge_index] == edge_b and relations[i][edge_index] in [
                        imp_relation_id, and_relation_id, or_relation_id]:
                        trans_exten_edge_ids[i].append(
                            [edge_a, graphs[i].edges()[1].numpy().tolist()[edge_index], relations[i][edge_index]])

        for _ in range(max_extension_depth - 1):
            if len(trans_exten_edge_ids[i]) > Config.extension_padding_len:
                break
            for relation_index in range(len(relations[i])):
                if relations[i][relation_index] == imp_relation_id:
                    edge_a, edge_b = graphs[i].edges()[0].numpy().tolist()[relation_index], \
                                     graphs[i].edges()[1].numpy().tolist()[relation_index]
                    for edge_index in range(len(graphs[i].edges()[0].numpy().tolist())):
                        if graphs[i].edges()[0].numpy().tolist()[edge_index] == edge_b and (
                                relations[i][edge_index] == imp_relation_id):
                            trans_exten_edge_ids[i].append(
                                [edge_a, graphs[i].edges()[1].numpy().tolist()[edge_index], imp_relation_id])
            edges_a, edges_b = graphs[i].edges()[0].numpy().tolist(), graphs[i].edges()[1].numpy().tolist()
            for index in range(len(trans_exten_edge_ids[i])):
                tmp_edges_id.append(len(edges_a))
                edges_a.append(trans_exten_edge_ids[i][index][0])
                edges_b.append(trans_exten_edge_ids[i][index][1])
                relations[i].append(trans_exten_edge_ids[i][index][2])
            graphs[i] = dgl.graph((edges_a, edges_b))
            assert is_dgl_graph_connected(graphs[i])

        for relation_index in range(len(relations[i])):
            if len(cont_exten_node_ids[i]) + len(trans_exten_edge_ids[i]) > Config.extension_padding_len:
                break
            if relations[i][relation_index] == imp_relation_id:
                edge_a, edge_b = graphs[i].edges()[0].numpy().tolist()[relation_index], \
                                 graphs[i].edges()[1].numpy().tolist()[relation_index]
                sentence_a = (node_sentences_a[i] + node_sentences_b[i])[edge_a].replace('_c', '').replace('_d', '')
                sentence_b = (node_sentences_a[i] + node_sentences_b[i])[edge_b].replace('_c', '').replace('_d', '')

                sentence_a_neg = get_from_new_not_sentence_map(sentence_a)
                sentence_b_neg = get_from_new_not_sentence_map(sentence_b)

                if len(sentence_a_neg) == 0 or len(sentence_b_neg) == 0 or sentence_a_neg[0] == 'None' or \
                        sentence_b_neg[
                            0] == 'None' or 'error when convert' in sentence_a_neg or 'error when convert' in sentence_b_neg:
                    continue
                not_a_node_id = add_node(sentence_a_neg[0] + ('_c' if edge_a < len(node_sentences_a[i]) else '_d'), i,
                                         'c' if edge_a < len(node_sentences_a[i]) else 'd')
                not_b_node_id = add_node(sentence_b_neg[0] + ('_c' if edge_b < len(node_sentences_a[i]) else '_d'), i,
                                         'c' if edge_b < len(node_sentences_a[i]) else 'd')

                add_edge(edge_a, not_a_node_id, 'NOT', i)
                add_edge(not_a_node_id, edge_a, 'NOT', i)
                add_edge(edge_b, not_b_node_id, 'NOT', i)
                add_edge(not_b_node_id, edge_b, 'NOT', i)
                add_edge(not_b_node_id, not_a_node_id, 'IMP', i)
                add_edge(not_a_node_id, not_b_node_id, 'IMP_REV', i)
                cont_exten_node_ids[i].append([edge_a, not_a_node_id, edge_b, not_b_node_id])

        edges_a, edges_b = graphs[i].edges()[0].numpy().tolist(), graphs[i].edges()[1].numpy().tolist()
        for edge_id in tmp_edges_id:
            edges_a[edge_id] = edges_b[edge_id] = relations[i][edge_id] = -1
        edges = (list(filter(lambda x: x != -1, edges_a)), list(filter(lambda x: x != -1, edges_b)))
        relations[i] = list(filter(lambda x: x != -1, relations[i]))

        new_edges, new_node_sentences_a, new_node_sentences_b, _base_node_id, _cont_exten_node_id, _trans_exten_edge_id = move_node_sentences(
            edges, node_sentences_a[i], node_sentences_b[i], node_sentences_c[i], node_sentences_d[i], base_node_ids[i],
            cont_exten_node_ids[i], trans_exten_edge_ids[i])
        base_node_ids[i] = _base_node_id
        cont_exten_node_ids[i] = _cont_exten_node_id
        trans_exten_edge_ids[i] = _trans_exten_edge_id
        node_sentences_a[i] = [s.replace('_c', '').strip() for s in new_node_sentences_a]
        node_sentences_b[i] = [s.replace('_d', '').strip() for s in new_node_sentences_b]
        graphs[i] = dgl.graph(new_edges)

    if return_base_nodes:
        return context, answers, graphs, node_sentences_a, node_sentences_b, relations, [get_edge_norm(r, g, g.edges())
                                                                                         for r, g in zip(relations,
                                                                                                         graphs)], base_node_ids, cont_exten_node_ids, trans_exten_edge_ids

    return context, answers, graphs, node_sentences_a, node_sentences_b, relations, [get_edge_norm(r, g, g.edges())
                                                                                     for r, g in
                                                                                     zip(relations, graphs)]


def is_dgl_graph_connected(graph, show_graph=False):
    ret = nx.is_connected(dgl.to_networkx(graph).to_undirected())
    return ret


def construct_logic_graph(id: str, return_base_nodes=False, min_edge_nums=Config.truncate_edges_num):
    context, answers, graphs, node_sentences_a, node_sentences_b, relations = construct_relation_graph_new(id=id, )

    for i in range(4):
        assert len(node_sentences_a[i]) + len(node_sentences_b[i]) == graphs[i].num_nodes()

    base_node_ids = None
    cont_exten_node_ids = None
    trans_exten_edge_ids = None
    for i in range(4):
        assert is_dgl_graph_connected(graphs[i])
    context, answers, graphs, node_sentences_a, node_sentences_b, relations, _, base_node_ids, cont_exten_node_ids, trans_exten_edge_ids = construct_relation_graph_extension(
        id=id, context=context, graphs=graphs, node_sentences_a=node_sentences_a, node_sentences_b=node_sentences_b,
        relations=relations, return_base_nodes=return_base_nodes)
    for i in range(4):
        assert is_dgl_graph_connected(graphs[i])

    for i in range(4):
        for j in range(len(trans_exten_edge_ids[i])):
            assert len(trans_exten_edge_ids[i][j]) == 3

    for i in range(4):
        count = 0
        while (graphs[i].num_nodes() > Config.truncate_nodes_num) and count < 3:
            context, answers, graphs, node_sentences_a, node_sentences_b, relations, _, merge_nodes = construct_relation_graph_merger_nodes(
                id=None, show_graph=False, graph_type=-1, context=context, answers=answers, graphs=graphs,
                _node_sentences_a=node_sentences_a, _node_sentences_b=node_sentences_b, relations=relations,
                selected_i=None, return_merge_nodes=True, base_node_ids=base_node_ids)
            count += 1
            base_node_ids, cont_exten_node_ids, trans_exten_edge_ids = tie_nodes_ids(base_node_ids, cont_exten_node_ids,
                                                                                     merge_nodes, trans_exten_edge_ids)
        assert is_dgl_graph_connected(graphs[i])
        assert len(node_sentences_a[i]) + len(node_sentences_b[i]) == graphs[i].num_nodes()
        if 'LogiQA' in dataset_dir:
            count = 0
            while graphs[i].num_nodes() > Config.truncate_nodes_num and count < 20:
                context, answers, graphs, node_sentences_a, node_sentences_b, relations, _, merge_nodes = construct_relation_graph_merger_nodes(
                    id=None, show_graph=False, graph_type=-1, context=context, answers=answers, graphs=graphs,
                    _node_sentences_a=node_sentences_a, _node_sentences_b=node_sentences_b, relations=relations,
                    selected_i=None, return_merge_nodes=True, forced_merge=True, base_node_ids=base_node_ids)
                count += 1
                base_node_ids, cont_exten_node_ids, trans_exten_edge_ids = tie_nodes_ids(base_node_ids,
                                                                                         cont_exten_node_ids,
                                                                                         merge_nodes,
                                                                                         trans_exten_edge_ids)
                assert len(node_sentences_a[i]) + len(node_sentences_b[i]) == graphs[i].num_nodes()
                if graphs[i].num_edges() > Config.truncate_edges_num:
                    context, answers, graphs, node_sentences_a, node_sentences_b, relations, _ = random_delete_edges(
                        context=context, answers=answers, graphs=graphs, _node_sentences_a=node_sentences_a,
                        _node_sentences_b=node_sentences_b, relations=relations, selected_i=i,
                        min_edge_nums=min_edge_nums)
                    assert len(node_sentences_a[i]) + len(node_sentences_b[i]) == graphs[i].num_nodes()
        if graphs[i].num_edges() > min_edge_nums:
            context, answers, graphs, node_sentences_a, node_sentences_b, relations, _ = random_delete_edges(
                context=context, answers=answers, graphs=graphs, _node_sentences_a=node_sentences_a,
                _node_sentences_b=node_sentences_b, relations=relations, selected_i=i, min_edge_nums=min_edge_nums)

        assert is_dgl_graph_connected(graphs[i]), id
        assert len(node_sentences_a[i]) + len(node_sentences_b[i]) == graphs[i].num_nodes()

    for i in range(4):
        assert is_dgl_graph_connected(graphs[i]), id
        edges_a, edges_b = graphs[i].edges()[0].numpy().tolist(), graphs[i].edges()[1].numpy().tolist()
        for index in range(len(edges_a)):
            if edges_a[index] == edges_b[index]:
                # self loop
                edges_a[index] = edges_b[index] = -1
                relations[i][index] = -1
        edges_a = list(filter(lambda x: x != -1, edges_a))
        edges_b = list(filter(lambda x: x != -1, edges_b))
        relations[i] = list(filter(lambda x: x != -1, relations[i]))
        graphs[i] = dgl.graph((edges_a, edges_b))
        assert is_dgl_graph_connected(graphs[i]), id
        assert len(node_sentences_a[i]) + len(node_sentences_b[i]) == graphs[i].num_nodes()

    for i in range(4):
        if len(trans_exten_edge_ids[i]) + len(cont_exten_node_ids) > Config.extension_padding_len:
            trans_exten_edge_ids[i] = trans_exten_edge_ids[i][:Config.extension_padding_len]
            cont_exten_node_ids[i] = cont_exten_node_ids[i][
                                     :Config.extension_padding_len - len(trans_exten_edge_ids[i])]
        node_sentences_a[i] = [sentence_format(s) for s in node_sentences_a[i]]
        node_sentences_b[i] = [sentence_format(s) for s in node_sentences_b[i]]
    return context, answers, graphs, node_sentences_a, node_sentences_b, relations, [get_edge_norm(r, g, g.edges()) for
                                                                                     r, g in zip(relations,
                                                                                                 graphs)], base_node_ids, cont_exten_node_ids, trans_exten_edge_ids


def tie_nodes_ids(base_node_ids, cont_exten_node_ids, merge_nodes, trans_exten_edge_ids):
    if base_node_ids is None:
        return base_node_ids, cont_exten_node_ids, trans_exten_edge_ids

    def larger_count(l: list, v: int):
        l = np.array(l)
        return l[l < v].size

    for i in range(4):
        for node in merge_nodes[i]:
            if node in base_node_ids[i]:
                base_node_ids[i][base_node_ids[i].index(node + 1)] = -1
        base_node_ids[i] = list(filter(lambda x: x != -1, base_node_ids[i]))

        for index in range(len(base_node_ids[i])):
            base_node_ids[i][index] -= larger_count(merge_nodes[i], base_node_ids[i][index])
        for index in range(len(cont_exten_node_ids[i])):
            for inner_index in range(4):
                cont_exten_node_ids[i][index][inner_index] -= larger_count(merge_nodes[i],
                                                                           cont_exten_node_ids[i][index][inner_index])
        for index in range(len(trans_exten_edge_ids[i])):
            for inner_index in range(2):
                trans_exten_edge_ids[i][index][inner_index] -= larger_count(merge_nodes[i],
                                                                            trans_exten_edge_ids[i][index][inner_index])

    return base_node_ids, cont_exten_node_ids, trans_exten_edge_ids
