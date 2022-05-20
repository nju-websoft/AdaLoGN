import logging
import os

import dgl
import torch
import torch.nn as nn
from dgl.nn.pytorch import GATConv

import Config
from utils import get_edge_norm

logger = logging.getLogger(__name__)


class ResidualGRU(nn.Module):
    def __init__(self, hidden_size, dropout, num_layers=2):
        super(ResidualGRU, self).__init__()
        self.enc_layer = nn.GRU(input_size=hidden_size, hidden_size=hidden_size // 2, num_layers=num_layers,
                                batch_first=True, dropout=dropout, bidirectional=True)
        self.enc_ln = nn.LayerNorm(hidden_size)

    def forward(self, input):
        self.enc_layer.flatten_parameters()
        output, _ = self.enc_layer(input)
        return self.enc_ln(output + input)


class GRUPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = ResidualGRU(hidden_size, dropout=Config.model_args.dropout)

    def forward(self, graphs, return_origin_shape=False):
        graphs_hidden_states = graphs.ndata['h']

        graphs_hidden_states_after_padding = []
        batch_num_nodes = graphs.batch_num_nodes().detach().cpu().numpy().tolist()
        max_node_num = max(batch_num_nodes)
        for index, num_nodes in enumerate(batch_num_nodes):
            start_index = sum(batch_num_nodes[:index])
            end_index = sum(batch_num_nodes[:index + 1])

            padding_embeddings = torch.zeros((max_node_num - num_nodes, self.hidden_size),
                                             dtype=graphs_hidden_states.dtype, device=graphs_hidden_states.device)
            graphs_hidden_states_after_padding.append(
                torch.cat([graphs_hidden_states[start_index:end_index], padding_embeddings], dim=0))

        graphs_hidden_states_after_padding = torch.stack(graphs_hidden_states_after_padding).view(-1, max_node_num,
                                                                                                  self.hidden_size)
        graphs_hidden_states = self.gru(graphs_hidden_states_after_padding)
        if not return_origin_shape:
            return graphs_hidden_states

        ret_graphs_hidden_states = []
        for index, num_nodes in enumerate(batch_num_nodes):
            ret_graphs_hidden_states.append(graphs_hidden_states[index, :num_nodes, :])
        return torch.cat(ret_graphs_hidden_states, dim=0)


class GraphAttentionPoolingWithGRU(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.graph_weight_pred = nn.Sequential(nn.Linear(hidden_size * 2, 1, bias=False), nn.LeakyReLU())
        self.gru_pooling = GRUPooling(hidden_size)

    def forward(self, graphs, attention_query):
        graphs_hidden_states = self.gru_pooling(graphs)
        max_node_num = max(graphs.batch_num_nodes().detach().cpu().numpy().tolist())
        graphs_node_weight = self.graph_weight_pred(torch.cat(
            [graphs_hidden_states, attention_query.view(-1, 1, self.hidden_size).repeat(1, max_node_num, 1)],
            dim=-1))
        graphs_node_weight = torch.softmax(graphs_node_weight, dim=1).view(len(graphs_hidden_states), max_node_num, 1)
        graphs_hidden_states = graphs_hidden_states * graphs_node_weight
        return torch.sum(graphs_hidden_states, dim=1).view(-1, self.hidden_size)


class feat_nn(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        assert len(x.shape) == 2
        return x[:, :x.shape[-1] // 2]


class GraphAttentionPooling(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.attention_layer = nn.Sequential(torch.nn.Linear(2 * hidden_size, 1, bias=False), torch.nn.LeakyReLU())
        self.graph_pooling = dgl.nn.GlobalAttentionPooling(self.attention_layer, feat_nn=feat_nn())

    def forward(self, graphs, attention_query):
        batch_num_nodes = graphs.batch_num_nodes().detach().cpu().numpy().tolist()
        attention_query = attention_query.view(len(batch_num_nodes), self.hidden_size)
        attention_query_repeat = []
        for index, num_nodes in enumerate(batch_num_nodes):
            attention_query_repeat.append(attention_query[index].view(1, self.hidden_size).repeat(num_nodes, 1))
        return self.graph_pooling(graphs,
                                  torch.cat([graphs.ndata['h'], torch.cat(attention_query_repeat, dim=0)], dim=-1))


class GraphSet2SetPooling(nn.Module):
    def __init__(self, hidden_size, set_2_set_n_iters=3, set_2_set_n_layers=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.graph_pooling = dgl.nn.Set2Set(hidden_size, set_2_set_n_iters, set_2_set_n_layers)
        self.output_layer = torch.nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, graphs, attention_query: None):
        return self.output_layer(self.graph_pooling(graphs, graphs.ndata['h']))


class GraphSortPooling(nn.Module):
    def __init__(self, hidden_size, k=5):
        super().__init__()
        self.hidden_size = hidden_size
        self.k = k
        self.graph_pooling = dgl.nn.SortPooling(k)
        self.attention_layer = nn.Sequential(nn.Linear(hidden_size * 2, 1, bias=False), nn.LeakyReLU())

    def forward(self, graphs, attention_query):
        output = self.graph_pooling(graphs, graphs.ndata['h']).view(graphs.batch_size, self.k, self.hidden_size)
        atten_weight = self.attention_layer(
            torch.cat([output, attention_query.view(graphs.batch_size, 1, self.hidden_size).repeat(1, self.k, 1)],
                      dim=-1)).view(graphs.batch_size, self.k, 1)
        atten_weight = torch.softmax(atten_weight, dim=1)
        return torch.sum(output * atten_weight, dim=1)


def nodes_with_feature_x(nodes, feature, x):
    return (nodes.data[feature] == x).view(-1)


def nodes_with_nodes_subgraph_type_0(nodes):
    return nodes_with_feature_x(nodes, 'subgraph_type', 0)


def nodes_with_nodes_subgraph_type_1(nodes):
    return nodes_with_feature_x(nodes, 'subgraph_type', 1)


class RGATLayer(nn.Module):
    def __init__(self, hidden_size, num_rels, num_bases=-1, bias=None, activation=None):
        super(RGATLayer, self).__init__()
        self.in_feat = hidden_size
        self.out_feat = hidden_size
        self.hidden_size = hidden_size
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.dropout = nn.Dropout(Config.model_args.dropout)
        self.subgraph_attn = nn.Sequential(nn.Linear(hidden_size * 2, 1, bias=False), nn.LeakyReLU())

        # sanity check
        if self.num_bases <= 0 or self.num_bases > self.num_rels:
            self.num_bases = self.num_rels
        # weight bases in equation (3)
        else:
            self.weight = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat, self.out_feat))
        self.subgraph_proj = nn.Linear(self.in_feat, self.out_feat)
        self.subgraph_gate = nn.Linear(self.in_feat * 2, 1, bias=False)
        if self.num_bases < self.num_rels:
            self.w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))
        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(hidden_size))
        self.attn_layer = nn.Sequential(torch.nn.Linear(2 * hidden_size, 1, bias=False), nn.LeakyReLU())
        self.extension_pred_layer = torch.nn.Linear(hidden_size * 2, 1)

        self.self_loop = torch.nn.Linear(hidden_size, hidden_size)

    def forward(self, _g, base_nodes_ids=None, exten_nodes_ids=None, exten_edges_ids=None, ):
        graphs_list = dgl.unbatch(_g)
        num_count = 0
        graphs_node_ids = [base_nodes_ids[i][base_nodes_ids[i] != Config.extension_padding_value] for i in
                           range(len(graphs_list))]
        rel_scores = []
        for i in range(len(graphs_list)):
            rel_socre_i = []
            exten_edges = [[], []]
            edge_relation_type = []
            for j in range(len(exten_edges_ids[i])):
                eni = exten_edges_ids[i][j][exten_edges_ids[i][j] != Config.extension_padding_value][:2]
                if len(eni) == 0:
                    continue
                pred_value = self.extension_pred_layer(torch.cat(
                    [torch.mean(torch.index_select(graphs_list[i].ndata['h'], dim=0, index=eni), dim=0),
                     torch.mean(torch.index_select(graphs_list[i].ndata['answer'], dim=0, index=eni), dim=0)],
                    dim=-1))
                pred_value = torch.sigmoid(pred_value).view(1)
                if pred_value > Config.extension_threshold:
                    exten_edges[0].append(eni[0].view(-1))
                    exten_edges[1].append(eni[1].view(-1))
                    edge_relation_type.append(exten_edges_ids[i][j][2].view(1))
                rel_socre_i.append(pred_value)
            if len(exten_edges[0]) != 0:
                graphs_list[i].add_edges(torch.cat(exten_edges[0]), torch.cat(exten_edges[1]),
                                         {'rel_type': torch.cat(edge_relation_type),
                                          'norm': torch.ones(len(exten_edges[0]), device=exten_edges_ids.device)})
            for j in range(len(exten_nodes_ids[i])):
                eni = exten_nodes_ids[i][j][exten_nodes_ids[i][j] != Config.extension_padding_value]
                if len(eni) == 0 or not graphs_list[i].has_edges_between(eni[0], eni[2]):
                    continue
                pred_value = self.extension_pred_layer(torch.cat(
                    [torch.mean(torch.index_select(graphs_list[i].ndata['h'], dim=0, index=eni), dim=0),
                     torch.mean(torch.index_select(graphs_list[i].ndata['answer'], dim=0, index=eni), dim=0)],
                    dim=-1))
                pred_value = torch.sigmoid(pred_value).view(1)
                if pred_value > Config.extension_threshold:
                    graphs_node_ids[i] = torch.unique(torch.cat([graphs_node_ids[i], eni]))
                rel_socre_i.append(pred_value)
            new_graph = dgl.node_subgraph(graphs_list[i], nodes=graphs_node_ids[i])
            new_graph.ndata['origin_id'] = graphs_node_ids[i] + num_count
            num_count += graphs_list[i].num_nodes()
            new_graph.edata['norm'] = torch.tensor(
                get_edge_norm(new_graph.edata['rel_type'], new_graph, new_graph.edges()),
                device=new_graph.device).view(-1)

            context_subgraph = dgl.node_subgraph(new_graph,
                                                 nodes=new_graph.filter_nodes(nodes_with_nodes_subgraph_type_0))
            choice_subgraph = dgl.node_subgraph(new_graph,
                                                nodes=new_graph.filter_nodes(nodes_with_nodes_subgraph_type_1))
            subgraph_message = []
            for index in range(new_graph.num_nodes()):
                subgraph = context_subgraph if new_graph.ndata['subgraph_type'][index] == 1 else choice_subgraph
                attn = self.subgraph_attn(torch.cat([subgraph.ndata['h'],
                                                     new_graph.ndata['h'][index].view(1,
                                                                                      self.hidden_size).repeat(
                                                         subgraph.num_nodes(), 1)], dim=-1))
                attn = torch.softmax(attn.view(-1), dim=-1)
                subgraph_message.append(torch.sum(subgraph.ndata['h'] * attn.view(-1, 1), dim=0))
            new_graph.ndata['subgraph_message'] = torch.cat(subgraph_message, dim=0).view(new_graph.num_nodes(),
                                                                                          self.hidden_size)
            graphs_list[i] = new_graph
            rel_scores.append(torch.mean(torch.cat(rel_socre_i, dim=-1)).view(1) if len(rel_socre_i) != 0 else
                              torch.ones(1, device=_g.device))
        g = dgl.batch(graphs_list)

        def edge_attn(edges):
            attn = self.attn_layer(torch.cat([edges.src['h'], edges.dst['h']], dim=-1))
            return {'e_attn': attn}

        def message_func(edges):
            w = self.weight[edges.data['rel_type'].cpu().numpy().tolist()]
            msg = torch.bmm(edges.src['h'].unsqueeze(1), w).squeeze()
            msg = msg * edges.data['norm'].view(-1, 1)
            return {'msg': msg, 'e_attn': edges.data['e_attn']}

        def reduce_func(nodes):
            alpha = torch.softmax(nodes.mailbox['e_attn'], dim=1)
            subgraph_gate_weight = self.subgraph_gate(
                torch.cat([nodes.data['h'], nodes.data['subgraph_message']], dim=-1))
            subgraph_gate_weight = torch.sigmoid(subgraph_gate_weight).view(-1, 1)
            subgraph_msg = self.subgraph_proj(nodes.data['subgraph_message']) * subgraph_gate_weight
            h = torch.sum(nodes.mailbox['msg'] * alpha, dim=1) + self.self_loop(nodes.data['h']) + subgraph_msg
            return {'h': h}

        def apply_func(nodes):
            h = nodes.data['h']
            if self.bias:
                h = h + self.bias
            return {'h': h}

        g.apply_edges(edge_attn)
        g.update_all(message_func, reduce_func, apply_func)
        new_data_map = dict(
            [nid.detach().cpu().numpy().tolist(), h] for nid, h in zip(g.ndata['origin_id'], g.ndata['h']))
        whole_g_h = _g.ndata['h']
        new_whole_g_h = []
        for i in range(len(whole_g_h)):
            if i in new_data_map:
                new_whole_g_h.append(new_data_map[i])
            else:
                new_whole_g_h.append(whole_g_h[i])
        h = torch.stack(new_whole_g_h)
        if self.activation is not None:
            h = self.activation(h)
        _g.ndata['h'] = h
        return self.dropout(h), g.ndata['origin_id'], g.batch_num_nodes(), g.batch_num_edges(), torch.cat(rel_scores,
                                                                                                          dim=-1).view(
            -1, 1)


class RGAT(nn.Module):
    def __init__(self, num_layers, hidden_dim, base_num, num_rels=6, activation=torch.relu):
        super(RGAT, self).__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(Config.model_args.dropout)
        self.rgat_layers = nn.ModuleList()
        self.global_pooling_layer = nn.Linear(hidden_dim * num_layers, hidden_dim)
        for _ in range(num_layers):
            self.rgat_layers.append(
                RGATLayer(hidden_dim, num_rels=num_rels, num_bases=base_num, activation=activation, ))
        pooling_type = Config.model_args.pooling_type
        pooling_classes = {'attention_pooling_with_gru': GraphAttentionPoolingWithGRU,
                           'attention_pooling': GraphAttentionPooling, 'set2set_pooling': GraphSet2SetPooling,
                           'sort_pooling': GraphSortPooling}
        assert pooling_type in pooling_classes
        self.graph_pooling = pooling_classes[pooling_type](hidden_dim)

    def forward(self, graph, base_nodes_ids, exten_nodes_ids, exten_edges_ids, attention_query, ):
        graph.ndata['origin_h'] = graph.ndata['h']
        all_rel_scores = []
        all_h = []
        for i in range(self.num_layers):
            h, node_ids, batch_num_nodes, batch_num_edges, rel_scores = self.rgat_layers[i](graph, base_nodes_ids,
                                                                                            exten_nodes_ids,
                                                                                            exten_edges_ids, )
            all_rel_scores.append(rel_scores)
            all_h.append(h)
        all_h = torch.cat(all_h, dim=-1)  # with size graph.num_nodes() * (d * num_layers)
        all_rel_scores = torch.cat(all_rel_scores, dim=-1)
        graph.ndata['h'] = all_h
        graph = dgl.node_subgraph(graph, nodes=node_ids)
        graph.set_batch_num_nodes(batch_num_nodes)
        graph.set_batch_num_edges(batch_num_edges)

        graph.ndata['h'] = self.global_pooling_layer(graph.ndata['h']) + graph.ndata['origin_h']

        output = torch.cat([self.graph_pooling(graph, attention_query), all_rel_scores], dim=-1)
        return output
