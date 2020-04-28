from subprocess import Popen, PIPE
from flask import Flask, Response, escape, request, render_template
import requests
import json

import subprocess

subprocess.Popen(["sh", "runSSTServer.sh"])
subprocess.Popen(["sh", "runStanfordParserServer.sh"])

process = Popen(['sh', 'run.sh'], stdin=PIPE, stdout=PIPE)
grep_stdout = process.communicate(input=b'Tim Cook is the CEO of Apple.\n')[0]
t = grep_stdout.decode()
questions = []
for m in t.split('\n'):
    question = m.split('\t')[0]
    if len(question) > 0:
        questions.append(question)


print('questions:', questions)

print('testing done.')


print('question generation test has finished.')


process = Popen(['sh', 'simplify.sh'], stdin=PIPE, stdout=PIPE)
grep_stdout = process.communicate(input=b'A ray, which is an idealized form of light, is represented by a wavelength.\n')[0]
t = grep_stdout.decode()
claims = [x for x in t.split('\n') if len(x) > 0]
print(claims)

print('sentence simplification done.')





app = Flask(__name__)

@app.route('/generate_question', methods=['GET', 'POST'])
def hello():


    if request.method == "POST":

        # get url that the user has entered
        try:
            claim = request.json['claim']
            claim_b = bytes(claim+'\n', encoding='ascii')
            process = Popen(['sh', 'run.sh'], stdin=PIPE, stdout=PIPE)
            grep_stdout = process.communicate(input=claim_b)[0]
            t = grep_stdout.decode()

            questions = []
            answers = []
            for m in t.split('\n'):
                x = m.split('\t')
                print('x:', x)
                question = x[0]
                if len(x) > 1:
                    answer = x[2]
                else:
                    answer = ''

                if len(question) > 0 and len(x) > 1:
                    questions.append(question)
                    answers.append(answer)


            print('questions:', questions)
            print('answers:', answers)
            print('question generation finished.')
            return Response(
                json.dumps({'questions': questions, 'answers': answers}),
                mimetype='application/json',
                headers={
                    'Cache-Control': 'no-cache',
                    'Access-Control-Allow-Origin': '*'
                }
            )
        except:
            print('Question generation error.')
            return Response(
                json.dumps({'error': 'question generation error'}),
                mimetype='application/json',
                headers={
                    'Cache-Control': 'no-cache',
                    'Access-Control-Allow-Origin': '*'
                }
            )





@app.route('/simplify_sentence', methods=['GET', 'POST'])
def simplify():

    if request.method == "POST":

        # get url that the user has entered
        try:
            claim = request.json['claim']
            claim_b = bytes(claim+'\n', encoding='ascii')
            process = Popen(['sh', 'simplify.sh'], stdin=PIPE, stdout=PIPE)
            grep_stdout = process.communicate(input=claim_b)[0]
            t = grep_stdout.decode()
            claims = []
            for m in t.split('\n'):
                claim = m.split('\t')[0]
                if len(claim) > 0:
                    claims.append(claim)

            print('claims:', claims)

            print('claim simplification finished.')
            return Response(
                json.dumps({'claims': claims}),
                mimetype='application/json',
                headers={
                    'Cache-Control': 'no-cache',
                    'Access-Control-Allow-Origin': '*'
                }
            )
        except:
            print('claim simplification error.')
            return Response(
                json.dumps({'error': 'claim simplification error'}),
                mimetype='application/json',
                headers={
                    'Cache-Control': 'no-cache',
                    'Access-Control-Allow-Origin': '*'
                }
            )

app.run(
    host='0.0.0.0',
    port=7000,
    debug=False,
    threaded=True
)
