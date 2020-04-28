import time

from pytorch_pretrained_bert.tokenization import BertTokenizer as GEARBertTokenizer
from pytorch_pretrained_bert.modeling import BertModel as GEARBertModel


from torch.utils.data import SequentialSampler


import collections

from models import GEAR
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset

import numpy as np


import unicodedata
import torch.nn.functional as F

from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm
import torch

from flask import Flask, Response, request
import json


app = Flask(__name__)



gear_tokenizer = GEARBertTokenizer.from_pretrained('../BERT-Pair/', do_lower_case=True)
bert_model = GEARBertModel.from_pretrained('../BERT-Pair/')

base_dir = '../gear-model/'
gear_model = GEAR(nfeat=768, nins=5, nclass=3, nlayer=1, pool='att')
optimizer = optim.Adam(gear_model.parameters(),
                       lr=0.005,
                       weight_decay=5e-4)
checkpoint = torch.load(base_dir + 'best.pth.tar', map_location=torch.device('cpu'))
gear_model.load_state_dict(checkpoint['model'])


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
            print('type vals, argmax', type(vals), type(argmax))
            argmax = int(argmax)
            return argmax, answer_list, vals

    print("--- %.3f seconds for gear inference ---" % (time.time() - start_time))


@app.route('/', methods=['GET', 'POST'])
def answer():
    claim = request.json['claim']
    evidences = request.json['evidences']
    argmax, evidences, vals = get_results_gear(claim, evidences)

    return Response(
                json.dumps({
                    'argmax': argmax,
                    'evidences': evidences,
                    'vals': vals,
                }),
                mimetype='application/json',
                headers={
                    'Cache-Control': 'no-cache',
                    'Access-Control-Allow-Origin': '*'
                }
            )



app.run(
        host='0.0.0.0',
        port=12000,
        debug=False,
        threaded=True
    )


