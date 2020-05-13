from subprocess import Popen, PIPE
from flask import Flask, Response, escape, request, render_template
import requests
import json
import tensorflow_hub as hub
from annoy import AnnoyIndex
import subprocess
import redis

print('getting USE model...')
embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder-large/5')
print('fetched model.')


r = redis.Redis(
    host='127.0.0.1',
    port=6379)

D=512
NUM_TREES=10
ann = AnnoyIndex(D, metric='angular')
ann.load('article_100.index')



app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def hello():

    query = request.json['claim']
    query_embedding = embed([query])
    nns = ann.get_nns_by_vector(query_embedding[0], 10)
    results = []
    for n in nns:
        result = json.loads(r.get(str(n)))
        results.append(result)

    return Response(
                    json.dumps({'news_article_evidences': results}),
                    mimetype='application/json',
                    headers={
                        'Cache-Control': 'no-cache',
                        'Access-Control-Allow-Origin': '*'
                    }
                )


app.run(
    host='0.0.0.0',
    port=15000,
    debug=False,
    threaded=True
)