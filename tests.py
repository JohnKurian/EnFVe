import requests
import json


#Testing GEAR model


claim = 'Al Jardine is an American rhythm guitarist.'
question = "Who is an american rhythm guitarist?"
evidences = [
    "He is best known as the band’s rhythm guitarist, and for occasionally singing lead vocals on singles such as “Help Me, Rhonda” (1965), “Then I Kissed Her” (1965) and “Come Go with Me” (1978).",
"Ray Jardine American rock climber, lightweight backpacker, inventor, author and global adventurer.",
    "Alan Charles Jardine (born September 3, 1942) is an American musician, singer and songwriter who co-founded the Beach Boys.",

    "In 1988, Jardine was inducted into the Rock and Roll Hall of Fame as a member of the Beach Boys.",
    "In 2010, Jardine released his debut solo studio album, A Postcard from California."
]


# claim = 'Vaccines cause autism'
# evidences = [
#     "The claim that vaccines causes autism has been disproven.",
#     "Vaccines do not cause autism"
# ]


#Testing GEAR model
# r = requests.post('http://0.0.0.0:12000/', json={'claim': claim, 'evidences': evidences})

#Testing Roberta model
# r = requests.post('http://0.0.0.0:20000/', json={'claim': claim, 'evidences': evidences})
# print(r.json())



def get_evidences(claim):
    r = requests.post('http://0.0.0.0:11000', json={'claim': claim})
    similar_lines = r.json()['similar_lines']
    similar_paras = r.json()['similar_paras']
    wiki_results_filtered = r.json()['wiki_results_filtered']
    img_urls = r.json()['img_urls']
    print('get evidences:', 'done')
    return similar_lines, similar_paras, wiki_results_filtered, img_urls



def coreference_resolution(claim):
    r = requests.post('http://0.0.0.0:8000', json={'claim': claim})
    resolved_coref = r.json()['resolved_coref']
    print('resolved coreference:', resolved_coref)
    return resolved_coref

def simplify_sentence(claim):
    r = requests.post('http://0.0.0.0:7000/simplify_sentence', json={'claim': claim})
    claims = r.json()['claims']
    print('simplify sentence:', claims)
    return claims

def generate_questions(claim):
    r = requests.post('http://0.0.0.0:7000/generate_question', json={'claim': claim})
    questions = r.json()['questions']
    gold_answers = r.json()['answers']
    print('generated questions:', questions)
    return questions, gold_answers

def retrieve_answer(question, evidences):
    r = requests.post('http://0.0.0.0:10000/', json={'question': question, 'evidences': evidences})
    answer = r.json()['answer']
    print('retrieve answers:', answer)
    return answer


def get_sentiment_analysis(claim):
    r = requests.post('http://0.0.0.0:9000/sentiment_analysis', json={'text': claim})
    return r.json()


def get_roberta_stances(claim, evidences):
    r = requests.post('http://0.0.0.0:14000/', json={'claim': claim, 'evidences': evidences})
    return r.json()['stances']


def get_results_gear_api(claim, evidences):
    r = requests.post('http://0.0.0.0:12000/', json={'claim': claim, 'evidences': evidences})
    argmax = r.json()['argmax']
    evidences =r.json()['evidences']
    vals = r.json()['vals']
    print('gear results:', r.json())
    return argmax, evidences, vals

def get_results_transformer_xh_api(claim, evidences):
r = requests.post('http://0.0.0.0:17000/', json={'claim': claim, 'evidences': evidences})
argmax = r.json()['argmax']
evidences =r.json()['evidences']
vals = r.json()['vals']
print('gear results:', r.json())
    return argmax, evidences, vals


def get_toxicity_scores(claim):
    api_key = 'AIzaSyBTrtgnaWeBx7Z-BMzVS3rL6REJFaaKxaM'
    url = ('https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze' +
           '?key=' + api_key)
    data_dict = {
        'comment': {'text': claim},
        'languages': ['en'],
        'requestedAttributes': {'TOXICITY': {},
                                'SEVERE_TOXICITY': {},
                                'IDENTITY_ATTACK': {},
                                'INSULT': {},
                                'PROFANITY': {},
                                'THREAT': {},
                                'SEXUALLY_EXPLICIT': {},
                                'FLIRTATION': {}}
    }
    response = requests.post(url=url, data=json.dumps(data_dict))
    response_dict = json.loads(response.content)
    attributes = list(response_dict['attributeScores'])
    scores = [item['summaryScore']['value'] for item in list(response_dict['attributeScores'].values())]

    toxicity_scores = {}
    for i in range(len(attributes)):
        toxicity_scores[attributes[i]] = round(scores[i], 2)
        print(f"{attributes[i]}: {round(scores[i], 2)}")

    return toxicity_scores


print(get_evidences(claim))
print(coreference_resolution(claim))
print(simplify_sentence(claim))
print(generate_questions(claim))
print(retrieve_answer(question, evidences))
print(get_roberta_stances(claim, evidences))
print(get_results_gear_api(claim, evidences))
print(get_results_transformer_xh_api(claim, evidences))
print(get_toxicity_scores(claim))

print('all api tests ran successfully.')
print('testing main backend server..')

def get_fact_check_prediction(claim):
    r = requests.post('http://0.0.0.0:5000/test', json={'claim': claim})
    return r.json()

claim = 'Lockheed Martin is developing a hybrid airship'
# evidences = get_evidences(claim)
print(get_fact_check_prediction(claim))


claim = 'norway has one of the smallest gaps between rich and poor'
print(get_fact_check_prediction(claim))
print('backend server test successful.')
