import json
from annoy import AnnoyIndex
from datetime import datetime
import subprocess
import redis
import json
import tensorflow as tf
import tensorflow_hub as hub


data = []

embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder-large/5')
print('fetched model.')

r = redis.Redis(
    host='127.0.0.1',
    port=6379)

D = 512
NUM_TREES = 10
ann = AnnoyIndex(D, metric='angular')
embedding_counter = 0

texts = []

with open('wiki/AA/wiki_00') as f:
    for line_index, line in enumerate(f):
        # print(line)
        embeddings = embed(line)
        print(embeddings)
        ann.add_item(line_index, embeddings[0])
        if line_index == 0:
            texts.append(line)
            break

        # data.append(json.loads(line))
        # ann.add_item(embedding_counter, e)
        # embedding_counter += 1
embeddings = embed(texts)

ann.build(NUM_TREES)
ann.save('wiki_articles.index')