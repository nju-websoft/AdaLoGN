import logging
import os

import dgl
import numpy as np
import torch
import torch.nn as nn
from transformers import RobertaModel
from transformers.modeling_outputs import MultipleChoiceModelOutput
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
import Config
from models.GNNs import RGAT

logger = logging.getLogger(__name__)


class LabelSmoothingCrossEntropyLoss(nn.Module):
    """NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.0):
        """Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        logger.info(f'label smoothing factor: {self.smoothing}')

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class RobertaAdaLoGN(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        model_type = Config.model_args.model_type
        gnn_layers_num = Config.model_args.gnn_layers_num

        logger.info(f'model type: {model_type}')
        logger.info(f'gnn layers num: {gnn_layers_num}')

        assert model_type == 'Roberta'
        self.roberta = RobertaModel(config)

        self.dropout = nn.Dropout(Config.model_args.dropout)


        self.output_layer1 = nn.Linear(config.hidden_size * 4 + Config.model_args.gnn_layers_num,
                                       config.hidden_size * 2)
        self.output_layer2 = nn.Linear(config.hidden_size * 2, 1)

        self.gnn = RGAT(gnn_layers_num, self.config.hidden_size, base_num=Config.model_args.base_num,
                        num_rels=Config.rgcn_relation_nums,)

        self.loss_fn = nn.CrossEntropyLoss() if not Config.model_args.label_smoothing else LabelSmoothingCrossEntropyLoss(
            smoothing=Config.model_args.label_smoothing_factor2)

        self.config = config

        self.padding_count = 0
        self.total_count = 0

        # self.init_weights()

    def _get_split_representation(self, input_ids, last_hidden_states, graph_batch_num, node_intervals,
                                  node_intervals_len):

        node_intervals = node_intervals.detach().cpu().numpy().tolist()
        node_intervals_len = node_intervals_len.detach().cpu().numpy().tolist()
        for i in range(len(node_intervals)):
            node_intervals[i] = node_intervals[i][:node_intervals_len[i][0]]

        node_representations = [torch.stack(
            [torch.mean(last_hidden_states[i, s:e, :], dim=0).view(self.config.hidden_size) for (s, e) in
             node_interval]) for i, node_interval in enumerate(node_intervals)]

        for i in range(len(node_representations)):
            if len(node_representations[i]) != graph_batch_num[i]:
                if len(node_representations[i]) > graph_batch_num[i]:
                    print((' '.join(Config.tokenizer.convert_ids_to_tokens(input_ids[i])).replace(
                        '[PAD]' if Config.model_type == 'Bert' else '<pad>', '').strip()).replace('Ġ', ''))
                    tokens = Config.tokenizer.convert_ids_to_tokens(input_ids[i])
                    tokens = list(map(lambda x: x.replace('Ġ', ''), tokens))
                    print(tokens)
                    print(f'i: {i}')
                    print(len(node_representations[i]))
                    print(graph_batch_num[i])
                    exit(-1)
                padding_representation = torch.zeros(self.config.hidden_size, device=input_ids.device)
                node_representations[i] = torch.cat([node_representations[i], torch.stack(
                    [padding_representation] * (graph_batch_num[i] - len(node_representations[i])))], dim=0)
                self.padding_count += 1
            self.total_count += 1
            assert len(node_representations[i]) == graph_batch_num[i]

        return node_representations

    def _get_split_origin_context_answer_representation(self, input_ids, last_hidden_states):
        sep_id = torch.tensor(Config.tokenizer.convert_tokens_to_ids(Config.SEP_TOKEN))
        # [CLS] context [SEP] question [SEP] answer [SEP]
        sep_locs = [np.where((input_id == sep_id).view(-1).detach().cpu().numpy())[0].tolist() for input_id in
                    input_ids]
        for i in range(len(sep_locs)):
            if sep_locs[i][1] - sep_locs[i][0] < 2:
                sep_locs[i] = [sep_locs[i][0], sep_locs[i][2]]
            if len(sep_locs[i]) == 2:
                sep_locs[i] = [sep_locs[i][0], sep_locs[i][0], sep_locs[i][1]]

        sep_interval = [[(1, sep_locs[index][0]), (sep_locs[index][0] + 1, sep_locs[index][1]),
                         (sep_locs[index][1] + 1, sep_locs[index][2])] for index in
                        range(len(sep_locs))]

        context_origin_representations = [
            torch.mean(last_hidden_states[index, si[0][0]:si[0][1], :], dim=0).view(self.config.hidden_size) for
            index, si in enumerate(sep_interval)]
        question_origin_representations = [torch.mean(last_hidden_states[index, si[1][0]:si[1][1], :], dim=0) for
                                           index, si in enumerate(sep_interval)]
        answer_origin_representations = [torch.mean(last_hidden_states[index, si[2][0]:si[2][1], :], dim=0) for
                                         index, si in enumerate(sep_interval)]
        return torch.stack(context_origin_representations), torch.stack(question_origin_representations), torch.stack(
            answer_origin_representations)

    def _get_dgl_graph_batch(self, input_ids, last_hidden_states, graphs: dgl.batch, node_intervals,
                             node_intervals_len, ):
        node_representations = self._get_split_representation(input_ids, last_hidden_states,
                                                              graphs.batch_num_nodes().detach().cpu().numpy().tolist(),
                                                              node_intervals, node_intervals_len, )
        graphs.ndata['h'] = torch.cat(node_representations, dim=0)
        return graphs

    def forward(
            self,
            input_ids,
            attention_mask,
            graphs,
            graph_node_nums,
            edge_types,
            edge_norms,
            labels, node_intervals, node_intervals_len,
            nodes_num, question_interval=None, context_interval=None, answer_interval=None,
            input_ids_origin=None,
            attention_mask_origin=None,
            token_type_ids_origin=None,
            token_type_ids=None,
            base_nodes_ids=None, exten_nodes_ids=None, exten_edges_ids=None
    ):

        input_ids = input_ids.view(-1, input_ids.size(-1))

        attention_mask = attention_mask.view(-1, attention_mask.size(-1))

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))

        input_ids_origin = input_ids_origin.view(-1, input_ids_origin.size(-1))
        attention_mask_origin = attention_mask_origin.view(-1, attention_mask_origin.size(-1))
        if token_type_ids_origin is not None:
            token_type_ids_origin = token_type_ids_origin.view(-1, token_type_ids_origin.size(-1))

        node_intervals = node_intervals.view(len(input_ids), Config.node_interval_padding_len, 2)
        node_intervals_len = node_intervals_len.view(len(input_ids), -1)

        base_nodes_ids = base_nodes_ids.view(len(input_ids), -1)
        exten_nodes_ids = exten_nodes_ids.view(len(input_ids), -1, 4)
        exten_edges_ids = exten_edges_ids.view(len(input_ids), -1, 3)

        nodes_num = nodes_num.view(len(input_ids), -1)
        device = f'cuda:{input_ids.get_device()}'
        r = torch.tensor(-1)
        graphs = dgl.batch(
            [dgl.graph((edge[0][edge[0] != r], edge[1][edge[1] != r]), num_nodes=graph_node_nums.view(-1)[index])
             for index, edge in enumerate(graphs.view(len(input_ids), 2, -1))]).to(device)

        nodes_subgraph_type = []
        graphs_batch_nodes_num = graphs.batch_num_nodes().detach().cpu().numpy().tolist()
        for index, _nodes_num in enumerate(graphs_batch_nodes_num):
            a_nodes_num, b_nodes_num = nodes_num[index].detach().cpu().numpy().tolist()
            nodes_subgraph_type += [0] * a_nodes_num
            nodes_subgraph_type += [1] * b_nodes_num

        graphs.ndata['subgraph_type'] = torch.tensor(nodes_subgraph_type, device=input_ids.get_device())
        graphs.edata['rel_type'] = torch.cat(
            [edge_type[edge_type != r].view(-1) for edge_type in edge_types.view(len(input_ids), -1)])
        graphs.edata['norm'] = torch.cat(
            [edge_norm[edge_norm != r].view(-1) for edge_norm in edge_norms.view(len(input_ids), -1)])
        bert_outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        bert_outputs_origin = self.roberta(
            input_ids_origin,
            attention_mask=attention_mask_origin,
            token_type_ids=token_type_ids_origin,
            return_dict=True,
        )
        last_hidden_states = bert_outputs['last_hidden_state']
        last_hidden_states = self.dropout(last_hidden_states)
        last_hidden_states_origin = bert_outputs_origin['last_hidden_state']
        last_hidden_states_origin = self.dropout(last_hidden_states_origin)

        context_origin_representations, question_origin_representations, answer_origin_representations = self._get_split_origin_context_answer_representation(
            input_ids_origin, last_hidden_states_origin)
        # self, input_ids, last_hidden_states, graphs: dgl.batch, question_interval, node_intervals, node_intervals_len,
        graphs = self._get_dgl_graph_batch(input_ids, last_hidden_states, graphs,
                                           node_intervals, node_intervals_len, )

        answer_origin_representations_for_graph = []
        for index, num_nodes in enumerate(graphs_batch_nodes_num):
            answer_origin_representations_for_graph.append(
                answer_origin_representations[index].view(1, self.config.hidden_size).repeat(num_nodes, 1))

        graphs.ndata['answer'] = torch.cat(answer_origin_representations_for_graph, dim=0)

        graphs_representations = self.gnn(graphs, base_nodes_ids, exten_nodes_ids, exten_edges_ids,
                                          attention_query=answer_origin_representations, )  # last_hidden_states_origin[:, 0, :])

        assert graphs_representations.shape == torch.Size(
            [len(input_ids), self.config.hidden_size + Config.model_args.gnn_layers_num])
        graphs_representations = self.dropout(graphs_representations)

        outputs = self.output_layer1(
            torch.cat([context_origin_representations, question_origin_representations, answer_origin_representations,
                       graphs_representations],
                      dim=-1))
        outputs = torch.tanh(outputs)
        outputs = self.output_layer2(outputs).view(-1, 4)
        loss = self.loss_fn(outputs, labels)
        torch.cuda.empty_cache()
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=outputs,
        )
