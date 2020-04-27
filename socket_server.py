from flask import Flask, escape, request, render_template, Response
import requests
import json
from flask_socketio import SocketIO, send


app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecret'

socketIo = SocketIO(app, cors_allowed_origins="*")

app.debug = True
app.host = '0.0.0.0'
app.port = '5000'



from flask import Flask, escape, request, render_template, Response
import requests

import time

from pytorch_pretrained_bert.tokenization import BertTokenizer as GEARBertTokenizer
from pytorch_pretrained_bert.modeling import BertModel as GEARBertModel

import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from tqdm import tqdm
import collections

from models import GEAR
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader

import numpy as np

import elasticsearch as es

import unicodedata

import argparse
import json
import os
import re
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import wikipedia
from allennlp.predictors.predictor import Predictor
from tqdm import tqdm

import tensorflow as tf
import tensorflow_hub as hub

from annoy import AnnoyIndex

import wikipediaapi

from spacy.lang.en import English

from serpwow.google_search_results import GoogleSearchResults
import json
# from pyfasttext import FastText

# from ESIM import ESIM


import torch.nn.functional as F
import dgl

from pytorch_transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

from data import HotpotDataset, FEVERDataset, TransformerXHDataset
from torch.utils.data import Dataset, DataLoader

from pytorch_transformers.tokenization_bert import BertTokenizer

from pytorch_transformers.modeling_bert import BertModel, BertEncoder, BertPreTrainedModel

import torch
import torch.nn as nn

import logging
from tqdm import tqdm

import json

import torch
from fairseq.data.data_utils import collate_tokens

from flask import jsonify

print('loading roberta model..')
roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')

from sentence_transformers import SentenceTransformer

print('loading distil bert siamese embedder...')
# embedder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

from scipy import spatial

algo = 'TXH'

# if True:
wiki_wiki = wikipediaapi.Wikipedia('en')

print('tf:')
tf.executing_eagerly()

module_url = "use_model"
module_url = 'http://tfhub.dev/google/universal-sentence-encoder/4'
use_model = hub.load(module_url)

test_sentences = [
    'A vaccine is a biological preparation that provides active acquired immunity to a particular infectious disease.',
    'A vaccine typically contains an agent that resembles a disease-causing microorganism and is often made from weakened or killed forms of the microbe, its toxins, or one of its surface proteins.']

print('testing USE model..')
print(use_model(test_sentences))

# USE emits 512 dimensional vectors
D = 512

# Default number of trees
NUM_TREES = 50

nltk.download('punkt')

predictor = Predictor.from_path(
    "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz")
# model = FastText("wiki.en.bin")

s = es.Elasticsearch([{'host': 'localhost', 'port': 9200}])


class InputExample(object):

    def __init__(self, unique_id, text_a, text_b, index, is_claim):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b
        # self.label = label
        self.index = index
        self.is_claim = is_claim


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids, index, is_claim):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids
        # self.label = label
        self.index = index
        self.is_claim = is_claim


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def read_from_json(input_json):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    unique_id = 0

    index = 0
    # label = input_json
    claim = input_json['claim']
    evidences = input_json['evidences']

    examples.append(InputExample(unique_id=unique_id, text_a=claim, text_b=None, index=index, is_claim=True))
    unique_id += 1

    for evidence in evidences:
        examples.append(InputExample(unique_id=unique_id, text_a=evidence, text_b=claim, index=index, is_claim=False))
        unique_id += 1
    return examples


def convert_examples_to_features(examples, seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]

        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length

        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids,
                index=example.index,
                is_claim=example.is_claim))
    return features


def feature_pooling_json(features):
    if len(features) == 0:
        features = []
        for i in range(5):
            features.append([0.0 for _ in range(768)])

    while len(features) < 5:
        features.append([0.0 for _ in range(len(features[0]))])

    return features


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

        print('details', logits_score, logits_pred, final_score)

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


gear_tokenizer = GEARBertTokenizer.from_pretrained('BERT-Pair/', do_lower_case=True)
bert_model = GEARBertModel.from_pretrained('BERT-Pair/')

base_dir = 'gear-model/'
gear_model = GEAR(nfeat=768, nins=5, nclass=3, nlayer=1, pool='att')
optimizer = optim.Adam(gear_model.parameters(),
                       lr=0.005,
                       weight_decay=5e-4)
checkpoint = torch.load(base_dir + 'best.pth.tar', map_location=torch.device('cpu'))
gear_model.load_state_dict(checkpoint['model'])

# c_gear_model = GEAR(nfeat=768, nins=5, nclass=2, nlayer=1, pool='att')
# optimizer = optim.Adam(c_gear_model.parameters(),
#                        lr=0.005,
#                        weight_decay=5e-4)
# checkpoint = torch.load(base_dir + 'contradiction_best.pth.tar', map_location=torch.device('cpu'))
# c_gear_model.load_state_dict(checkpoint['model'])


tokenizer = BertTokenizer.from_pretrained(config["bert_token_file"])

model = Model_FEVER(config)
# model.half()
model.network.to(torch.device("cpu"))

model.load('transformer_xh_model/model_finetuned_epoch_0.pt')
config = json.load(open('configs/config_fever.json', 'r', encoding="utf-8"))


def doc_retrieval(claim):
    # claim = 'Modi is the president of India'

    # claim = 'Modi is the president of India in 1992'

    def get_NP(tree, nps):

        if isinstance(tree, dict):
            if "children" not in tree:
                if tree['nodeType'] == "NP":
                    # print(tree['word'])
                    # print(tree)
                    nps.append(tree['word'])
            elif "children" in tree:
                if tree['nodeType'] == "NP":
                    # print(tree['word'])
                    nps.append(tree['word'])
                    get_NP(tree['children'], nps)
                else:
                    get_NP(tree['children'], nps)
        elif isinstance(tree, list):
            for sub_tree in tree:
                get_NP(sub_tree, nps)

        return nps

    def get_subjects(tree):
        subject_words = []
        subjects = []
        for subtree in tree['children']:
            if subtree['nodeType'] == "VP" or subtree['nodeType'] == 'S' or subtree['nodeType'] == 'VBZ':
                subjects.append(' '.join(subject_words))
                subject_words.append(subtree['word'])
            else:
                subject_words.append(subtree['word'])
        return subjects

    # predictor.predict(claim)
    tokens = predictor.predict(claim)
    nps = []
    tree = tokens['hierplane_tree']['root']
    # print(tree)
    noun_phrases = get_NP(tree, nps)

    subjects = get_subjects(tree)
    for subject in subjects:
        if len(subject) > 0:
            noun_phrases.append(subject)
    # noun_phrases = list(set(noun_phrases))

    predicted_pages = []
    if len(noun_phrases) == 1:
        for np in noun_phrases:
            if len(np) > 300:
                continue
            docs = wikipedia.search(np)

            predicted_pages.extend(docs[:2])  # threshold

    else:
        for np in noun_phrases:
            if len(np) > 300:
                continue
            docs = wikipedia.search(np)

            predicted_pages.extend(docs[:1])

    wiki_results = set(predicted_pages)

    # wiki_results = []
    # for page in predicted_pages:
    #     page = page.replace(" ", "_")
    #     page = page.replace("(", "-LRB-")
    #     page = page.replace(")", "-RRB-")
    #     page = page.replace(":", "-COLON-")
    #     wiki_results.append(page)
    # print(wiki_results)

    noun_phrases = set(noun_phrases)
    f_predicted_pages = []
    for np in noun_phrases:
        page = np.replace('( ', '-LRB-')
        page = page.replace(' )', '-RRB-')
        page = page.replace(' - ', '-')
        page = page.replace(' -', '-')
        page = page.replace(' :', '-COLON-')
        page = page.replace(' ,', ',')
        page = page.replace(" 's", "'s")
        page = page.replace(' ', '_')

        if len(page) < 1:
            continue
        f_predicted_pages.append(page)

    noun_phrases = list(set(noun_phrases))

    wiki_results = list(set(wiki_results))

    # stop_words = set(stopwords.words('english'))
    # wiki_results = [w for w in wiki_results if not w in stop_words]

    claim = normalize(claim)
    claim = claim.replace(".", "")
    claim = claim.replace("-", " ")
    proter_stemm = nltk.PorterStemmer()
    tokenizer = nltk.word_tokenize
    words = [proter_stemm.stem(word.lower()) for word in tokenizer(claim)]
    words = set(words)

    for page in wiki_results:
        page = normalize(page)
        processed_page = re.sub("-LRB-.*?-RRB-", "", page)
        processed_page = re.sub("_", " ", processed_page)
        processed_page = re.sub("-COLON-", ":", processed_page)
        processed_page = processed_page.replace("-", " ")
        processed_page = processed_page.replace("–", " ")
        processed_page = processed_page.replace(".", "")
        page_words = [proter_stemm.stem(word.lower()) for word in tokenizer(processed_page) if
                      len(word) > 0]

        if all([item in words for item in page_words]):
            if ':' in page:
                page = page.replace(":", "-COLON-")
            f_predicted_pages.append(normalize(page))
    f_predicted_pages = list(set(f_predicted_pages))

    print(f'Search Entities: {noun_phrases}')
    print(f'Articles Retrieved: {wiki_results}')
    print(f'Predicted Retrievals: {f_predicted_pages}')

    filtered_lines = []
    print('looping...')
    print(wiki_results)

    wiki_results_list = []
    for result in wiki_results:
        try:
            p = wiki_wiki.page(result).text
            lines = p.split('\n')
            for line in lines:
                line.replace('\\', '')
                if not line.startswith('==') and len(line) > 60:
                    line = nltk.sent_tokenize(line)
                    for l in line:
                        wiki_results_list.append(result)
                    filtered_lines.extend(line)
        except:
            print('error')

    print('filtered_lines:', filtered_lines)

    # for result in wiki_results:
    #     try:
    #         p = wiki_wiki.page(result).text
    #         # p = p.replace('\n', ' ')
    #         # p = p.replace('\t', ' ')
    #         # filtered_lines = nltk.sent_tokenize(p)
    #         # filtered_lines = [line for line in filtered_lines if not line.startswith('==') and len(line) > 10 ]
    #
    #         # Load English tokenizer, tagger, parser, NER and word vectors
    #         nlp = English()
    #         # Create the pipeline 'sentencizer' component
    #         sbd = nlp.create_pipe('sentencizer')
    #         # Add the component to the pipeline
    #         nlp.add_pipe(sbd)
    #         text = p
    #         #  "nlp" Object is used to create documents with linguistic annotations.
    #         doc = nlp(text)
    #         # create list of sentence tokens
    #         filtered_lines = []
    #         for sent in doc.sents:
    #             txt = sent.text
    #             # txt = txt.replace('\n', '')
    #             # txt = txt.replace('\t', '')
    #             filtered_lines.append(txt)
    #
    #     #     lines = p.split('\n')
    #     #     for line in lines:
    #     #         line.replace('\\', '')
    #     #         if not line.startswith('==') and len(line) > 60:
    #     #             line = nltk.sent_tokenize(line)
    #     #             filtered_lines.extend(line)
    #     except:
    #         print('error')

    print('filtered_lines:', len(filtered_lines))
    return filtered_lines, wiki_results_list


def doc_retrieval_only_pages(claim):
    def get_NP(tree, nps):

        if isinstance(tree, dict):
            if "children" not in tree:
                if tree['nodeType'] == "NP":
                    # print(tree['word'])
                    # print(tree)
                    nps.append(tree['word'])
            elif "children" in tree:
                if tree['nodeType'] == "NP":
                    # print(tree['word'])
                    nps.append(tree['word'])
                    get_NP(tree['children'], nps)
                else:
                    get_NP(tree['children'], nps)
        elif isinstance(tree, list):
            for sub_tree in tree:
                get_NP(sub_tree, nps)

        return nps

    def get_subjects(tree):
        subject_words = []
        subjects = []
        for subtree in tree['children']:
            if subtree['nodeType'] == "VP" or subtree['nodeType'] == 'S' or subtree['nodeType'] == 'VBZ':
                subjects.append(' '.join(subject_words))
                subject_words.append(subtree['word'])
            else:
                subject_words.append(subtree['word'])
        return subjects

    tokens = predictor.predict(claim)
    nps = []
    tree = tokens['hierplane_tree']['root']
    noun_phrases = get_NP(tree, nps)

    subjects = get_subjects(tree)
    for subject in subjects:
        if len(subject) > 0:
            noun_phrases.append(subject)

    predicted_pages = []
    if len(noun_phrases) == 1:
        for np in noun_phrases:
            if len(np) > 300:
                continue
            docs = wikipedia.search(np)
            predicted_pages.extend(docs[:2])  # threshold

    else:
        for np in noun_phrases:
            if len(np) > 300:
                continue
            docs = wikipedia.search(np)
            predicted_pages.extend(docs[:1])

    wiki_results = list(predicted_pages)

    return wiki_results


def get_evidences_es(claim):
    res = s.search(index="wikipedia_en", size=5, body={"query": {
        "match": {
            "text": {
                "query": claim
            }
        }}})

    print("Got %d Hits:" % res['hits']['total']['value'])
    search_list = []
    result_list = []
    answer_list = []
    for hit in res['hits']['hits']:
        answer_list.append(hit["_source"]['text'])
        print(hit["_source"]['text'])
        print('\n\n')
    return answer_list


def get_evidences_siamese_distilbert(claim):
    # Corpus with example sentences
    evidences = doc_retrieval(claim)
    evidence_embeddings = embedder.encode(evidences)

    claim_embedding = embedder.encode(claim)

    distances = spatial.distance.cdist(claim_embedding, evidence_embeddings, "cosine")

    ranked_sentences = [evidences[index] for index in list(distances[0].argsort())]
    ranked_sentences = ranked_sentences[:5]
    print(ranked_sentences)

    return ranked_sentences

    # query_embeddings = embedder.encode(queries)


def get_evidences_use_annoy(claim):
    filtered_lines, wiki_results = doc_retrieval(claim)

    # filtered_lines = [x for x in filtered_lines if len(x)>60]

    print('getting embeddings..')
    embeddings = use_model(filtered_lines)
    print(embeddings)

    claim_embedding = use_model([claim])

    print('numpy distance..')

    distances = [np.linalg.norm(a - claim_embedding) for a in embeddings]
    idxs = np.argsort(distances)[::-1][-5:]
    nns = idxs.tolist()

    nns.reverse()

    print('wiki results:', wiki_results)

    wiki_results_filtered = [wiki_results[n] for n in nns ]

    paras = [[x - 1, x, x + 1] for x in nns]

    print('the claim:', claim)
    for idx, i in enumerate(nns):
        if i == 0:
            paras[idx] = [0, 1, 2]
        elif i == len(filtered_lines) - 1:
            paras[len(nns) - 1] = [len(filtered_lines) - 3, len(filtered_lines) - 2, len(filtered_lines) - 1]




        # ann = AnnoyIndex(D)
    #
    # for index, embed in enumerate(embeddings):
    #     ann.add_item(index, embed)
    # ann.build(NUM_TREES)
    #
    # nns = ann.get_nns_by_vector(claim_embedding[0], 5)

    print('paras:', paras)
    print('nns:', nns)
    similar_lines = [filtered_lines[i] for i in nns]
    similar_paras = [[filtered_lines[x] for x in para_idxs] for para_idxs in paras]
    print('similar lines:', similar_lines)
    print('similar paras:', similar_paras)
    return similar_lines, similar_paras, wiki_results_filtered


def get_evidences_esim(claim):
    documents = doc_retrieval_only_pages(claim)

    def get_words(claims, sents):

        words = set()
        for claim in claims:
            # print(claim)
            for idx, word in enumerate(nltk.word_tokenize(claim)):
                if idx >= 20:
                    break
                words.add(word.lower())
                # print(words)
        for sent in sents:
            # print(sent)
            for idx, word in enumerate(nltk.word_tokenize(sent)):
                if idx >= 20:
                    break
                words.add(word.lower())
                # print(words)
        return words

    def get_predict_words(devs):

        dev_words = set()

        for dev in tqdm(devs):
            claims = set()
            sents = []
            # for pair in dev:
            claims.add(dev[0])
            sents.append(dev[1])
            dev_tokens = get_words(claims, sents)
            dev_words.update(dev_tokens)
        print("words processing done!")
        return dev_words

    def word_2_dict(words):

        word_dict = {}
        for idx, word in enumerate(words):
            word = word.replace('\n', '')
            word = word.replace('\t', '')
            word_dict[idx] = word

        return word_dict

    def get_complete_words(dev_data):

        all_words = set()
        dev_words = get_predict_words(dev_data)
        # print(dev_words)
        all_words.update(dev_words)
        # print(all_words)
        word_dict = word_2_dict(all_words)
        print("Word Dict created")
        return word_dict

    def inverse_word_dict(word_dict):

        iword_dict = {}
        for key, word in tqdm(word_dict.items()):
            iword_dict[word] = key
        return iword_dict

    def sent_2_index(sent, i_word_dict, max_length):
        words = nltk.word_tokenize(sent)
        word_indexes = []
        for idx, word in enumerate(words):
            if idx >= max_length:
                break
            else:
                word_indexes.append(i_word_dict[word.lower()])
        return word_indexes

    def predict_data_indexes(data, word_dict):

        test_indexes = []

        sent_indexes = []
        claim = tests[0][0]
        claim_index = sent_2_index(claim, word_dict, 20)
        claim_indexes = [claim_index] * len(data)

        for claim, sent in data:
            sent_index = sent_2_index(sent, word_dict, 20)
            sent_indexes.append(sent_index)

        assert len(sent_indexes) == len(claim_indexes)
        test_indexes = list(zip(claim_indexes, sent_indexes))
        test_indexes.append(test_indexes)
        print("Indexes created")
        return test_indexes[:-1]

    def embed_to_numpy(embed_dict):

        feat_size = len(embed_dict[list(embed_dict.keys())[0]])
        embed = np.zeros((len(embed_dict) + 1, feat_size), np.float32)
        for k in embed_dict:
            embed[k] = np.asarray(embed_dict[k])
        print('Generate numpy embed:', embed.shape)

        return embed

    def softmax(prediction):
        theta = 2.0
        ps = np.exp(prediction * theta)
        ps /= np.sum(ps)
        return ps

    def averaging(predictions):
        processed_predictions = []
        for prediction in predictions:
            prediction = np.asarray(prediction)
            prediction = softmax(prediction)
            processed_predictions.append(prediction)
        processed_predictions = np.asarray(processed_predictions)
        final_prediction = np.mean(processed_predictions, axis=0, keepdims=False)

        return final_prediction

    def scores_processing(all_predictions):
        ensembled_predictions = []
        for i in range(len(all_predictions[0])):
            predictions = []
            for j in range(len(all_predictions)):
                predictions.append(all_predictions[j][i])
                # print(predictions)
            ensembled_prediction = averaging(predictions)
            ensembled_predictions.append(ensembled_prediction)
        return ensembled_predictions

    wiki_wiki = wikipediaapi.Wikipedia(language='en', extract_format=wikipediaapi.ExtractFormat.WIKI)

    tests = []
    test_location_indexes = []
    pages = set(documents)
    p_lines = []
    doc_lines = []
    for page in pages:
        if wiki_wiki.page(page).exists():
            p = wiki_wiki.page(page).text
            lines = p.split('\n')
            for line in lines:
                line.replace('\\', '')
                if not line.startswith('==') and len(line) > 60:
                    line = nltk.sent_tokenize(line)
                    doc_lines.extend(line)

            if not doc_lines:
                return []
            document_lines = list(zip(doc_lines, [page] * len(doc_lines), range(len(doc_lines))))
            p_lines.extend(document_lines)
        # print(p_lines)

    for doc_line in p_lines:
        if not doc_line[0]:
            continue
        tests.append((claim, doc_line[0]))

        test_location_indexes.append((doc_line[1], doc_line[2]))

    if len(tests) == 0:
        tests.append((claim, 'no evidence for this claim'))
        test_location_indexes.append(('empty', 0))

    word_dict = get_complete_words(tests)
    iword_dict = inverse_word_dict(word_dict)

    _PAD_ = len(word_dict)
    word_dict[_PAD_] = '[PAD]'
    iword_dict['[PAD]'] = _PAD_
    test_indexes = predict_data_indexes(tests, iword_dict)

    embed_dict = {}

    for word, key in iword_dict.items():
        embed_dict[key] = model[word]

    print('Embedding size: %d' % (len(embed_dict)))

    embed = embed_to_numpy(embed_dict)

    clf = ESIM(h_max_length=20, s_max_length=20, learning_rate=0.001, batch_size=256, num_epoch=20,
               model_store_dir="/esim_pretrained_model", embedding=embed, word_dict=iword_dict,
               dropout_rate=0.2, random_state=88, num_units=128, activation=tf.nn.relu, share_rnn=True)

    clf.restore_model("best_model.ckpt")

    all_predictions = []
    predictions = []
    # for test_index in tqdm(test_indexes):
    prediction = clf.predict(test_indexes)
    predictions.append(prediction)
    all_predictions.append(predictions)

    ensembled_predicitons = scores_processing(all_predictions)

    tf.reset_default_graph()

    ep = [value[0] for value in ensembled_predicitons[0]]
    idx = (-np.array(ep)).argsort()[:50]
    evidences = [tests[i] for i in idx]
    evidences = [evidence[1] for evidence in evidences]
    evidences = list(dict.fromkeys(evidences))
    print("\nEvidences")
    return evidences[:5]


def get_evidences_serpwow(claim):
    # location

    snippets = []
    trailing_cleaned_snippets = []
    date_removed_snippets = []
    # create the serpwow object, passing in our API key
    serpwow = GoogleSearchResults("671C61C0DE2642A29AFD9A89D0DF4075")

    # set up a dict for the search parameters
    params = {
        "q": claim,
    }

    # retrieve the search results as JSON
    result = serpwow.get_json(params)
    # with open('serpwow.json', 'a') as json_file:
    for organic_result in result['organic_results']:
        snippets.append(organic_result['snippet'])
    for snippet in snippets:
        s = snippet.replace('... ', '')
        s = s.replace('\xa0...', '')
        trailing_cleaned_snippets.append(s)

    # print(trailing_cleaned_snippets)
    for snippet in trailing_cleaned_snippets:
        if snippet[12] == '-':
            # print('date exists')
            # print(snippet[14:])
            date_removed_snippets.append(snippet[14:])

        elif snippet[13] == '-':
            # print('date exists')
            # print(snippet[15:])
            date_removed_snippets.append(snippet[15:])
        else:
            date_removed_snippets.append(snippet)

    print('serpwow results:', date_removed_snippets[:5])
    return date_removed_snippets[:5]


def get_results_transformer_xh(claim, evidences):
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


def get_results_gear(claim, answer_list):
    print('--- gear prediction phase ---')

    input_data = {
        'claim': claim,
        'evidences': answer_list
    }

    print('input data:', input_data)

    # input_data = {
    #     'claim': 'Jim is born in Brazil.',
    #     'evidences': ['Jim is born in Canada.', "He moved to Brazil at the age of 13.", "He moved to Brazil at the age of 13.", "He moved to Brazil at the age of 13."]
    # }

    start_time = time.time()

    examples = read_from_json(input_data)

    features = convert_examples_to_features(
        examples=examples, seq_length=128, tokenizer=gear_tokenizer)

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.input_type_ids for f in features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_example_index)

    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=1)

    sentence_embeddings = []
    for input_ids, input_mask, segment_ids, example_indices in tqdm(eval_dataloader, desc="testing"):
        with torch.no_grad():
            _, pooled_output = bert_model(input_ids, token_type_ids=segment_ids, attention_mask=input_mask,
                                          output_all_encoded_layers=False)

        sentence_embeddings.extend(pooled_output.detach().cpu().numpy())
    # all_label = [f.label for f in features]
    all_index = [f.index for f in features]
    all_is_claim = [f.is_claim for f in features]
    # print(len(sentence_embeddings[0]))

    instances = {}
    for i in range(len(sentence_embeddings)):
        # label = all_label[i]
        index = all_index[i]
        is_claim = all_is_claim[i]
        # embedding = sentence_embeddings[i].detach().cpu().numpy()
        embedding = sentence_embeddings[i]

        if index not in instances:
            instances[index] = {}
            # instances[index]['label'] = label
            if is_claim:
                instances[index]['claim'] = embedding
                instances[index]['evidences'] = []
            else:
                instances[index]['evidences'] = [embedding]
        else:
            # assert instances[index]['label'] == label
            if 'evidences' not in instances[index]:
                instances[index]['evidences'] = []
            instances[index]['evidences'].append(embedding)

    for instance in instances.items():
        output_json = collections.OrderedDict()
        output_json['index'] = instance[0]
        # output_json['label'] = instance[1]['label']
        output_json['claim'] = torch.tensor([[round(x.item(), 6) for x in instance[1]['claim']]])
        evidences = []
        for evidence in instance[1]['evidences']:
            item = [round(x.item(), 6) for x in evidence]
            evidences.append(item)
        evidences = feature_pooling_json(evidences)
        # print(evidences)
        output_json['evidences'] = torch.tensor([evidences])
        # print('output json:', output_json)

    dev_features = output_json.get('evidences')
    dev_claims = output_json.get('claim')
    dev_features, dev_claims = Variable(dev_features), Variable(dev_claims)
    dev_data = TensorDataset(dev_features, dev_claims)
    dev_dataloader = DataLoader(dev_data, batch_size=1)

    print("--- %.3f seconds to finish bert encoding ---" % (time.time() - start_time))

    start_time = time.time()

    dev_tqdm_iterator = tqdm(dev_dataloader)
    with torch.no_grad():
        for index, data in enumerate(dev_tqdm_iterator):
            feature_batch, claim_batch = data
            print('feature_batch:', feature_batch.shape)
            print('claim_batch:', claim_batch.shape)
            outputs = gear_model(feature_batch, claim_batch)
            # c_outputs = c_gear_model(feature_batch, claim_batch)
            print('outputs:', outputs)
            # print('c_outputs:', c_outputs)
            vals = [abs(1 / x) for x in np.array(outputs).tolist()[0]]
            argmax = np.argmax(outputs.numpy())
            print('v:', argmax)
            return argmax, answer_list, vals

    print("--- %.3f seconds for gear inference ---" % (time.time() - start_time))


def get_stances_ucnlp(claim, evidences):
    return requests.post('http://127.0.0.1:6000', json={'claim': claim, 'evidences': evidences})


def get_roberta_preds(claim, evidences):
    batch_of_pairs = [[claim, evidence] for evidence in evidences]

    # batch_of_pairs = [
    #     ['Roberta is a heavily optimized version of BERT.', 'Roberta is not very optimized.'],
    #     ['Roberta is a heavily optimized version of BERT.', 'Roberta is based on BERT.'],
    #     ['potatoes are awesome.', 'I like to run.'],
    #     ['Mars is very far from earth.', 'Mars is very close.'],
    #     ['Roberta is a heavily optimized version of BERT.', 'Roberta is not very optimized.'],
    #     ['Roberta is a heavily optimized version of BERT.', 'Roberta is based on BERT.'],
    #     ['potatoes are awesome.', 'I like to run.'],
    #     ['Mars is very far from earth.', 'Mars is very close.'],
    #     ['Roberta is a heavily optimized version of BERT.', 'Roberta is not very optimized.'],
    #     ['Roberta is a heavily optimized version of BERT.', 'Roberta is based on BERT.']
    # ]

    batch = collate_tokens(
        [roberta.encode(pair[0], pair[1]) for pair in batch_of_pairs], pad_idx=1
    )

    logprobs = roberta.predict('mnli', batch)

    pred_dict = {
        0: 'contradiction',
        1: 'neutral',
        2: 'entailment'
    }
    pred_indices = logprobs.argmax(dim=1).tolist()
    preds = [pred_dict[i] for i in pred_indices]
    print(preds)
    return preds





@app.route('/', methods=['GET', 'POST'])
def hello():
    errors = []
    evidences = []
    vals = []
    results = {}
    prediction_result = ''
    claim = ''
    gear_result_names = ['True', 'Refutes', 'Not enough info']
    txh_result_names = ['Not enough info', 'refutes', 'supports']

    txh_argmax = [2, 1, 0]
    print('something')
    d = {'status': 'something'}
    print('request obj:', request)
    print('json:', request.json)
    response = jsonify({'some': 'data'})
    response.headers.add('Access-Control-Allow-Origin', '*')


    claim = request.form['url']
    print(claim)
    evidences = get_evidences_use_annoy(claim)
    # stances = get_stances_ucnlp(claim, evidences)
    # stances = stances.json()
    # roberta_preds = get_roberta_preds(claim, evidences)
    argmax, evidences, vals = get_results_gear(claim, evidences)
    print('gear results:', argmax, vals)
    # argmax, evidences, vals = get_results_transformer_xh(claim, evidences)
    # argmax = [txh_argmax[i] for i in argmax]
    # print('transformer xh results:', argmax, vals)
    # print('unc results:', stances)
    # print('roberta results:', roberta_preds)
    prediction_result = gear_result_names[argmax]


    return Response(
        json.dumps(d),
        mimetype='application/json',
        headers={
            'Cache-Control': 'no-cache',
            'Access-Control-Allow-Origin': '*'
        }
    )






@socketIo.on("message")
def handleMessage(msg):
    print('something')
    print(msg)

    gear_result_names = ['True', 'Refutes', 'Not enough info']
    claim = msg
    print(claim)
    evidences, paras, wiki_results = get_evidences_use_annoy(claim)
    argmax, evidences, vals = get_results_gear(claim, evidences)
    print('gear results:', argmax, vals)
    prediction_result = gear_result_names[argmax]

    paras_joined = [' '.join(para) for para in paras]

    d = {
        'prediction_result': prediction_result,
        'pred_vals': vals,
        'evidences': evidences,
        'paras': paras,
        'wiki_results': wiki_results,
        'paras_joined': paras_joined
    }

    send(d, broadcast=True)
    return None


socketIo.run(app)

