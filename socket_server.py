from flask_socketio import SocketIO, send
from flask import Flask, Response
import requests
import json


app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecret'

socketIo = SocketIO(app, cors_allowed_origins="*")

app.debug = True
app.host = '0.0.0.0'
app.port = '5000'



def get_stances_ucnlp(claim, evidences):
    return requests.post('http://127.0.0.1:6000', json={'claim': claim, 'evidences': evidences})


def get_evidences(claim):
    r = requests.post('http://127.0.0.1:11000', json={'claim': claim})
    similar_lines = r.json()['similar_lines']
    similar_paras = r.json()['similar_paras']
    wiki_results_filtered = r.json()['wiki_results_filtered']
    img_urls = r.json()['img_urls']
    print('get evidences:', 'done')
    return similar_lines, similar_paras, wiki_results_filtered, img_urls



def coreference_resolution(claim):
    r = requests.post('http://127.0.0.1:8000', json={'claim': claim})
    resolved_coref = r.json()['resolved_coref']
    print('resolved coreference:', resolved_coref)
    return resolved_coref

def simplify_sentence(claim):
    r = requests.post('http://127.0.0.1:7000/simplify_sentence', json={'claim': claim})
    claims = r.json()['claims']
    print('simplify sentence:', claims)
    return claims

def generate_questions(claim):
    r = requests.post('http://127.0.0.1:7000/generate_question', json={'claim': claim})
    questions = r.json()['questions']
    answers = r.json()['answers']
    print('generate questions:', answers)
    return questions, answers

def retrieve_answers(question, evidences):
    r = requests.post('http://127.0.0.1:10000/', json={'question': question, 'evidences': evidences})
    answer = r.json()['answer']
    print('retrieve answers:', answer)
    return answer


def get_sentiment_analysis(claim):
    r = requests.post('http://127.0.0.1:9000/sentiment_analysis', json={'text': claim})
    return r.json()

def get_results_gear_api(claim, evidences):
    r = requests.post('http://127.0.0.1:12000/', json={'claim': claim, 'evidences': evidences})
    argmax = r.json()['argmax']
    evidences =r.json()['evidences']
    vals = r.json()['vals']
    print('gear results:', r.json())
    return argmax, evidences, vals



@socketIo.on("message")
def handleMessage(msg):
    print('something')
    print(msg)

    gear_result_names = ['True', 'Refutes', 'Not enough info']
    claim = msg
    print(claim)

    claim = coreference_resolution(claim)
    claims = simplify_sentence(claim)


    for claim in claims:
        evidences, paras, wiki_results, img_urls = get_evidences(claim)
        argmax, evidences, vals = get_results_gear_api(claim, evidences)
        print('gear results:', argmax, vals)
        prediction_result = gear_result_names[argmax]

        paras_joined = [' '.join(para) for para in paras]

        d = {
            'prediction_result': prediction_result,
            'pred_vals': vals,
            'evidences': evidences,
            'paras': paras,
            'wiki_results': wiki_results,
            'paras_joined': paras_joined,
            'img_urls': img_urls
        }
        print('sending data..')
        send(d, broadcast=True)
        print('done.')
    return None


@app.route('/', methods=['GET', 'POST'])
def answer():
    return Response(
                json.dumps({'key': 'value'}),
                mimetype='application/json',
                headers={
                    'Cache-Control': 'no-cache',
                    'Access-Control-Allow-Origin': '*'
                }
            )


socketIo.run(app)

