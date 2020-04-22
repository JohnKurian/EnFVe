
from flask import Flask, jsonify, Response, escape, request, render_template
import requests
import json

import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer

from SentimentAnalysis.model import SentimentClassifier

# To test the POST request
#  r = requests.post('http://127.0.0.1:9000/sentiment_analysis', json={'text': 'I hate icecream'})
# r.json()




net = SentimentClassifier()
#CPU or GPU
device = torch.device("cpu")
#Put the network to the GPU if available
net = net.to(device)
#Instantiate the bert tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#Load the state dictionary of the network
net.load_state_dict(torch.load('SentimentAnalysis/models/model_bert', map_location=device))


app = Flask(__name__)

def classify_sentiment(sentence):
	with torch.no_grad():
		tokens = tokenizer.tokenize(sentence)
		tokens = ['[CLS]'] + tokens + ['[SEP]']
		tokens_ids = tokenizer.convert_tokens_to_ids(tokens)
		seq = torch.tensor(tokens_ids)
		seq = seq.unsqueeze(0)
		attn_mask = (seq != 0).long()
		logit = net(seq, attn_mask)
		prob = torch.sigmoid(logit.unsqueeze(-1))
		prob = prob.item()
		soft_prob = prob > 0.5
		if soft_prob == 1:
			return 'Positive', int(prob*100)
		else:
			return 'Negative', int(100-prob*100)



@app.route('/', methods=['GET', 'POST'])
def base():
    return 'hi'

@app.route('/sentiment_analysis', methods=['GET', 'POST'])
def sentiment_analyis():
	text = request.json['text']
	print('incoming text:', text)
	print('running sentiment classifier...')
	sentiment, probability = classify_sentiment(text)
	return jsonify({'sentiment': sentiment, 'probability': probability})

app.run(
        host='0.0.0.0',
        port=9000,
        debug=False,
        threaded=True
    )
