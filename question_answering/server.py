from allennlp.predictors.predictor import Predictor
from flask import Flask, Response, escape, request, render_template
import json

print('loading qa_predictor...')
qa_predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/bidaf-model-2017.09.15-charpad.tar.gz")


app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def answer():
    question = request.json['question']
    evidences = request.json['evidences']
    passage = ' '.join(evidences)
    p = qa_predictor.predict(
        passage=passage,
        question=question)

    return Response(
                json.dumps({'answer': p['best_span_str']}),
                mimetype='application/json',
                headers={
                    'Cache-Control': 'no-cache',
                    'Access-Control-Allow-Origin': '*'
                }
            )



app.run(
        host='0.0.0.0',
        port=10000,
        debug=False,
        threaded=True
    )
