from flask_socketio import SocketIO, send
from flask import Flask, Response
import requests

import numpy as np


import unicodedata
import tensorflow as tf
import torch.nn.functional as F
import dgl

from pytorch_transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

# from data import FEVERDataset
from torch.utils.data import DataLoader
from pytorch_transformers.modeling_bert import BertModel, BertEncoder, BertPreTrainedModel
from pytorch_pretrained_bert.tokenization import BertTokenizer
import torch.nn as nn
import logging
from tqdm import tqdm
import json
import torch

from flask import Flask, Response, request
import json


app = Flask(__name__)

def normalize(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', text)


'''
Graph Attention network component
'''


class GraphAttention(nn.Module):
    def __init__(self, in_dim=64,
                 out_dim=64,
                 num_heads=12,
                 feat_drop=0.6,
                 attn_drop=0.6,
                 alpha=0.2,
                 residual=True):

        super(GraphAttention, self).__init__()
        self.num_heads = num_heads
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        if feat_drop:
            self.feat_drop = nn.Dropout(feat_drop)
        else:
            self.feat_drop = lambda x: x
        if attn_drop:
            self.attn_drop = nn.Dropout(attn_drop)
        else:
            self.attn_drop = lambda x: x
        self.attn_l = nn.Parameter(torch.Tensor(size=(1, num_heads, out_dim)))
        self.attn_r = nn.Parameter(torch.Tensor(size=(1, num_heads, out_dim)))
        nn.init.xavier_normal_(self.fc.weight.data, gain=1.414)
        nn.init.xavier_normal_(self.attn_l.data, gain=1.414)
        nn.init.xavier_normal_(self.attn_r.data, gain=1.414)
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.residual = residual
        self.activation = nn.ReLU()
        self.res_fc = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.xavier_normal_(self.res_fc.weight.data, gain=1.414)

    ### this is Gragh attention network part, we follow standard inplementation from DGL library
    def forward(self, g):
        self.g = g
        h = g.ndata['h']
        h = h.reshape((h.shape[0], self.num_heads, -1))
        ft = self.fc(h)
        a1 = (ft * self.attn_l).sum(dim=-1).unsqueeze(-1)
        a2 = (ft * self.attn_r).sum(dim=-1).unsqueeze(-1)
        g.ndata.update({'ft': ft, 'a1': a1, 'a2': a2})
        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self.reduce_func)
        ret = g.ndata['ft']
        if self.residual:
            if self.res_fc is not None:
                resval = self.res_fc(h)
            else:
                resval = torch.unsqueeze(h, 1)
            ret = resval + ret
        g.ndata['h'] = self.activation(ret.flatten(1))

    def message_func(self, edges):
        return {'z': edges.src['ft'], 'a': edges.data['a']}

    def reduce_func(self, nodes):
        alpha = F.softmax(nodes.mailbox['a'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'ft': h}

    def edge_attention(self, edges):
        a = self.leaky_relu(edges.src['a1'] + edges.dst['a2'])
        return {'a': a}


'''
Transformer-XH Encoder, we apply on last three BERT layers 

'''


class TransformerXHEncoder(BertEncoder):
    def __init__(self, config):
        super(TransformerXHEncoder, self).__init__(config)
        self.heads = ([8] * 1) + [1]
        self.config = config
        self.build_model()
        ### Here we apply on the last three layers, but it's ok to try different layers here.
        self.linear_layer1 = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.linear_layer2 = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.linear_layer3 = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.linear_layer1.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        self.linear_layer2.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        self.linear_layer3.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

    def build_model(self):
        self.graph_layers = nn.ModuleList()
        # input to hidden
        device = torch.device("cpu")

        i2h = self.build_input_layer().to(device)
        self.graph_layers.append(i2h)
        # hidden to hidden
        h2h = self.build_hidden_layer().to(device)
        self.graph_layers.append(h2h)
        h2h = self.build_hidden_layer().to(device)
        self.graph_layers.append(h2h)

    ### here the graph has dimension 64, with 12 heads, the dropout rates are 0.6
    def build_input_layer(self):
        return GraphAttention()

    def build_hidden_layer(self):
        return GraphAttention()

    def forward(self, graph, hidden_states, attention_mask, gnn_layer_num, output_all_encoded_layers=True):
        all_encoder_layers = []
        for i, layer_module in enumerate(self.layer):
            hidden_states = layer_module(hidden_states, attention_mask)[0]
            pooled_output = hidden_states[:, 0]
            graph.ndata['h'] = pooled_output

            if i >= gnn_layer_num:
                if i == 9:
                    g_layer = self.graph_layers[0]
                    g_layer(graph)
                    graph_outputs = graph.ndata.pop('h')
                    ht_ori = hidden_states.clone()
                    ht_ori[:, 0] = self.linear_layer1(torch.cat((graph_outputs, pooled_output), -1))
                elif i == 10:
                    g_layer = self.graph_layers[1]
                    g_layer(graph)
                    graph_outputs = graph.ndata.pop('h')
                    ht_ori = hidden_states.clone()
                    ht_ori[:, 0] = self.linear_layer2(torch.cat((graph_outputs, pooled_output), -1))
                else:
                    g_layer = self.graph_layers[2]
                    g_layer(graph)
                    graph_outputs = graph.ndata.pop('h')
                    ht_ori = hidden_states.clone()
                    ht_ori[:, 0] = self.linear_layer3(torch.cat((graph_outputs, pooled_output), -1))
                hidden_states = ht_ori
                if output_all_encoded_layers:
                    all_encoder_layers.append(ht_ori)
            else:
                if output_all_encoded_layers:
                    all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class Transformer_xh(BertModel):
    def __init__(self, config):
        super(Transformer_xh, self).__init__(config)

        self.encoder = TransformerXHEncoder(config)

    def forward(self, graph, input_ids, token_type_ids=None, attention_mask=None, position_ids=None, gnn_layer=11):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
        encoder_outputs = self.encoder(graph, embedding_output,
                                       extended_attention_mask, gnn_layer)
        sequence_output = encoder_outputs[-1]
        pooled_output = self.pooler(sequence_output)
        outputs = sequence_output, pooled_output  # add hidden_states and attentions if they are here

        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


class ModelHelper(BertPreTrainedModel):
    def __init__(self, node_encoder: BertModel, bert_config, config_model):
        super(ModelHelper, self).__init__(bert_config)
        ### node_encoder -> Transformer-XH
        self.node_encoder = node_encoder
        self.config_model = config_model
        self.node_dropout = nn.Dropout(self.config.hidden_dropout_prob)
        self.final_layer = nn.Linear(self.config.hidden_size, 1)
        self.final_layer.apply(self.init_weights)
        # self.init_weights()

    def forward(self, batch, device):
        pass


class Model:
    def __init__(self, config):
        self.config = config
        self.config_model = config['model']
        self.bert_node_encoder = Transformer_xh.from_pretrained(self.config['bert_model_file'],
                                                                cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(
                                                                    0))
        self.bert_config = self.bert_node_encoder.config
        self.network = ModelHelper(self.bert_node_encoder, self.bert_config, self.config_model)
        self.device = torch.device("cpu")

    def half(self):
        self.network.half()

    def eval(self):
        self.network.eval()

    def train(self):
        self.network.train()

    def save(self, filename: str):
        network = self.network
        if isinstance(network, nn.DataParallel):
            network = network.module

        return torch.save(self.network.state_dict(), filename)

    def load(self, model_state_dict: str):
        return self.network.load_state_dict(torch.load(model_state_dict, map_location=lambda storage, loc: storage))


class ModelHelper_FEVER(ModelHelper):
    def __init__(self, node_encoder, bert_config, config_model):
        super(ModelHelper_FEVER, self).__init__(node_encoder, bert_config, config_model)
        self.pred_final_layer = nn.Linear(self.config.hidden_size, 3)
        self.pred_final_layer.apply(self.init_weights)
        # self.init_weights()

    def forward(self, batch, device):
        ### Transformer-XH for node representations
        g = batch[0]
        g.ndata['encoding'] = g.ndata['encoding'].to(device)
        g.ndata['encoding_mask'] = g.ndata['encoding_mask'].to(device)
        g.ndata['segment_id'] = g.ndata['segment_id'].to(device)
        outputs = self.node_encoder(g, g.ndata['encoding'], g.ndata['segment_id'], g.ndata['encoding_mask'],
                                    gnn_layer=self.config_model['gnn_layer'])
        node_sequence_output = outputs[0]
        node_pooled_output = outputs[1]
        node_pooled_output = self.node_dropout(node_pooled_output)

        #### Task specific layer (last layer)
        logits_score = self.final_layer(node_pooled_output).squeeze(-1)
        logits_pred = self.pred_final_layer(node_pooled_output)

        return logits_score, logits_pred


class Model_FEVER(Model):
    def __init__(self, config):
        super(Model_FEVER, self).__init__(config)
        self.network = ModelHelper_FEVER(self.bert_node_encoder, self.bert_config, self.config_model)


config = {'name': 'transformer_xh_fever', 'task': 'fever',
          'model': {'bert-model': True, 'vocab_size': 40000, 'emb_size': 200, 'bert_max_len': 130, 'gnn_layer': 9,
                    'cuda_device': 1}, 'training': {
        'optimizer': {'type': 'Adam', 'params': {'lr': 0.004, 'amsgrad': True}, 'scheduler': {'type': 'CyclicLR',
                                                                                              'params': {
                                                                                                  'base_lr': 0.0001,
                                                                                                  'max_lr': 0.004,
                                                                                                  'step_size': 5,
                                                                                                  'mode': 'triangular'}}},
        'epochs': 2, 'train_batch_size': 1, 'test_batch_size': 1, 'num_workers': 0,
        'gradient_clipping': {'use': False, 'clip_value': 1.0}, 'shuffle': True, 'warmup_proportion': 0,
        'learning_rate': 1e-05, 'decay_rate': 0.99, 'decay_step': 100000, 'total_training_steps': 500000},
          'system': {'device': 'cuda', 'num_workers': 1, 'base_dir': 'experiments/',
                     'train_data': 'data/fever_train_graph.json', 'validation_data': 'data/fever_dev_graph.json',
                     'test_data': 'data/fever_dev_graph.json'}, 'bert_token_file': 'bert-base-uncased',
          'bert_model_file': 'bert-base-uncased',
          'bert_model_config': {'attention_probs_dropout_prob': 0.1, 'directionality': 'bidi', 'hidden_act': 'gelu',
                                'hidden_dropout_prob': 0.1, 'hidden_size': 768, 'initializer_range': 0.02,
                                'intermediate_size': 3072, 'max_position_embeddings': 512, 'num_attention_heads': 12,
                                'num_hidden_layers': 12, 'pooler_fc_size': 768, 'pooler_num_attention_heads': 12,
                                'pooler_num_fc_layers': 3, 'pooler_size_per_head': 128,
                                'pooler_type': 'first_token_transform', 'dropout': 0.6, 'gnn_hidden_state': 64,
                                'gnn_head': 12}}
config = {'name': 'transformer_xh_fever', 'task': 'fever',
          'model': {'bert-model': True, 'vocab_size': 40000, 'emb_size': 200, 'bert_max_len': 130, 'gnn_layer': 9},
          'training': {'optimizer': {'type': 'Adam', 'params': {'lr': 0.004, 'amsgrad': True},
                                     'scheduler': {'type': 'CyclicLR',
                                                   'params': {'base_lr': 0.0001, 'max_lr': 0.004, 'step_size': 5,
                                                              'mode': 'triangular'}}}, 'epochs': 2,
                       'train_batch_size': 1, 'test_batch_size': 1, 'num_workers': 0,
                       'gradient_clipping': {'use': False, 'clip_value': 1.0}, 'shuffle': True, 'warmup_proportion': 0,
                       'learning_rate': 1e-05, 'decay_rate': 0.99, 'decay_step': 100000,
                       'total_training_steps': 500000},
          'system': {'device': 'cpu', 'num_workers': 1, 'base_dir': 'experiments/',
                     'train_data': 'data/fever_train_graph.json', 'validation_data': 'data/fever_dev_graph.json',
                     'test_data': 'data/fever_dev_graph.json'}, 'bert_token_file': 'bert-base-uncased',
          'bert_model_file': 'bert-base-uncased',
          'bert_model_config': {'attention_probs_dropout_prob': 0.1, 'directionality': 'bidi', 'hidden_act': 'gelu',
                                'hidden_dropout_prob': 0.1, 'hidden_size': 768, 'initializer_range': 0.02,
                                'intermediate_size': 3072, 'max_position_embeddings': 512, 'num_attention_heads': 12,
                                'num_hidden_layers': 12, 'pooler_fc_size': 768, 'pooler_num_attention_heads': 12,
                                'pooler_num_fc_layers': 3, 'pooler_size_per_head': 128,
                                'pooler_type': 'first_token_transform', 'dropout': 0.6, 'gnn_hidden_state': 64,
                                'gnn_head': 12}}


def evaluation_fever(model, config, tokenizer, json_obj, inference=False):
    dataset = FEVERDataset(json_obj, config["model"], False, tokenizer)
    dataloader = DataLoader(dataset=dataset,
                            batch_size=config['training']['test_batch_size'],
                            collate_fn=batcher_fever(torch.device("cpu")),
                            shuffle=False,
                            num_workers=0)
    logging.info("=================================================================================")
    total = 0
    count = 0
    pred_dict = dict()
    logits_score = []
    final_score = []

    for batch in tqdm(dataloader):
        print('batch:', batch)

        logits_score, logits_pred = model.network(batch, torch.device("cpu"))

        logits_score = F.softmax(logits_score)
        logits_pred = F.softmax(logits_pred, dim=1)
        final_score = torch.mm(logits_score.unsqueeze(0), logits_pred).squeeze(0)


        # details
        # tensor([9.8728e-01, 6.9391e-04, 1.5247e-03, 2.4105e-03, 8.0916e-03],
        #        grad_fn= < SoftmaxBackward >) tensor([[2.0125e-04, 1.3085e-05, 9.9979e-01],
        #                                              [5.4027e-01, 2.4709e-03, 4.5726e-01],
        #                                              [1.9579e-01, 2.3639e-03, 8.0185e-01],
        #                                              [1.0890e-01, 1.6875e-03, 8.8942e-01],
        #                                              [2.8175e-02, 2.7256e-04, 9.7155e-01]],
        #                                             grad_fn= < SoftmaxBackward >) tensor(
        #     [1.3626e-03, 2.4511e-05, 9.9861e-01], grad_fn= < SqueezeBackward1 >)

        values, index = final_score.topk(1)

        print('final score:', index[0].item())

        final_score = final_score.tolist()

        pred_dict = index[0].item()

    return pred_dict, logits_score, final_score


def batcher_fever(device):
    def batcher_dev(batch):
        batch_graphs = dgl.batch([batch[0][0]])
        batch_graphs.ndata['encoding'] = batch_graphs.ndata['encoding'].to(device)
        batch_graphs.ndata['encoding_mask'] = batch_graphs.ndata['encoding_mask'].to(device)
        batch_graphs.ndata['segment_id'] = batch_graphs.ndata['segment_id'].to(device)
        qid = [batch[0][1]]
        label = [batch[0][2]]

        return batch_graphs, batch_graphs.ndata['label'].to(device), torch.tensor(label, dtype=torch.long).to(
            device), qid

    return batcher_dev



tokenizer = BertTokenizer.from_pretrained(config["bert_token_file"])

model = Model_FEVER(config)
# model.half()
model.network.to(torch.device("cpu"))

model.load('transformer_xh_model/model_finetuned_epoch_0.pt')
config = json.load(open('configs/config_fever.json', 'r', encoding="utf-8"))


@app.route('/', methods=['GET', 'POST'])
def get_results_transformer_xh():
    claim = request.json['claim']
    evidences = request.json['evidences']
    json_obj = {}
    json_obj['qid'] = 0
    json_obj['label'] = 'SUPPORTS'
    json_obj['question'] = claim

    node_list = []
    for i, evidence in enumerate(evidences):
        temp = {}
        temp['node_id'] = i
        temp['name'] = 'title_' + str(i)
        temp['context'] = tokenizer.tokenize(evidence)
        temp['label'] = 0
        temp['sent_num'] = i
        node_list.append(temp)

    json_obj['node'] = node_list

    json_batch = [json_obj]

    print('json batch:', json_batch)

    print('im also here')
    pred_dict, logits_score, final_score = evaluation_fever(model, config, tokenizer, json_batch, False)
    logits_score = logits_score.tolist()

    argmax = np.argmax(final_score)

    return argmax, evidences, final_score


app.run(
        host='0.0.0.0',
        port=17000,
        debug=False,
        threaded=True
    )