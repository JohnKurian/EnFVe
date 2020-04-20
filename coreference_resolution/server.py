# Load your usual SpaCy model (one of SpaCy English models)
# https://github.com/huggingface/neuralcoref

#virtualenv coref
#pip install -r requirements
#python -m spacy download en
#Then finall sh setup.sh


from subprocess import Popen, PIPE
from flask import Flask, Response, escape, request, render_template
import requests
import json


import spacy
nlp = spacy.load('en')

# Add neural coref to SpaCy's pipe
import neuralcoref
neuralcoref.add_to_pipe(nlp)



app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def coref():

    if request.method == "POST":

        # get url that the user has entered
        try:
            claim = request.json['claim']
            # You're done. You can now use NeuralCoref as you usually manipulate a SpaCy document annotations.
            doc = nlp(claim)
            resolved_coref = doc._.coref_resolved
            print("Resolved by NeuralCoref: \n")
            print(resolved_coref)
            return Response(
                json.dumps({'resolved_coref': resolved_coref}),
                mimetype='application/json',
                headers={
                    'Cache-Control': 'no-cache',
                    'Access-Control-Allow-Origin': '*'
                }
            )
        except:
            print('coref error.')
            return Response(
                json.dumps({'error': 'coref error'}),
                mimetype='application/json',
                headers={
                    'Cache-Control': 'no-cache',
                    'Access-Control-Allow-Origin': '*'
                }
            )

app.run(
        host='0.0.0.0',
        port=8000,
        debug=False,
        threaded=True
    )
