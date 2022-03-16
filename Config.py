import os

model_type = os.environ['MODEL_TYPE']
assert model_type == 'Roberta'
SEP_TOKEN = '</s>'
CLS_TOKEN = '<s>'
NODE_SEP_TOKEN = '|'
tokenizer = None
eval_test = True
model_args = None
rgcn_relation_nums = -1
node_intervals_padding_id = 10086
node_interval_padding_len = 50
extension_padding_value = 10086
extension_padding_len = 3
extension_threshold = -1
truncate_edges_num = 50
truncate_nodes_num = 25

max_edge_num = 101
max_node_num = 50
