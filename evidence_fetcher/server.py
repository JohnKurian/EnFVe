from allennlp.predictors.predictor import Predictor
from flask import Flask, Response, escape, request, render_template
import json


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

# from serpwow.google_search_results import GoogleSearchResults
import json
# from pyfasttext import FastText

# from ESIM import ESIM

from scipy import spatial
import requests

import unicodedata
import numpy as np

import elasticsearch as es

print('loading evidence fetcher...')



algo = 'TXH'

s = es.Elasticsearch([{'host': 'localhost', 'port': 9200}])

# if True:
wiki_wiki = wikipediaapi.Wikipedia('en')

import torch
import torch.nn as nn

import logging
from tqdm import tqdm

import json

import torch
from fairseq.data.data_utils import collate_tokens

from flask import jsonify

print('loading roberta model..')
# roberta = torch.hub.load('pytorch/fairseq', 'roberta.large.mnli')

from sentence_transformers import SentenceTransformer

print('loading distil bert siamese embedder...')
# embedder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

from scipy import spatial

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


predictor = Predictor.from_path(
    "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz")
# model = FastText("wiki.en.bin")

app = Flask(__name__)



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


def normalize(text):
    """Resolve different type of unicode encodings."""
    return unicodedata.normalize('NFD', text)


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
def get_evidences_use_annoy():
    print('req:', request.json)
    claim = request.json['claim']
    filtered_lines, wiki_results = doc_retrieval(claim)

    # filtered_lines = [x for x in filtered_lines if len(x)>60]

    print('getting embeddings..')
    embeddings = use_model(filtered_lines)

    claim_embedding = use_model([claim])

    print('numpy distance..')

    distances = [np.linalg.norm(a - claim_embedding) for a in embeddings]
    idxs = np.argsort(distances)[::-1][-5:]
    nns = idxs.tolist()

    nns.reverse()


    wiki_results_filtered = [wiki_results[n] for n in nns ]

    paras = [[x - 1, x, x + 1] for x in nns]

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

    similar_lines = [filtered_lines[i] for i in nns]
    similar_paras = [[filtered_lines[x] for x in para_idxs] for para_idxs in paras]

    img_urls = []
    for result in wiki_results_filtered:
        base_url = "https://en.wikipedia.org/w/api.php?action=query&prop=pageimages&format=json&piprop=original&titles=" + result
        r = requests.get(base_url)
        img_url = r.json()['query']['pages'][list(r.json()['query']['pages'].keys())[0]]['original']['source']
        img_urls.append(img_url)

    return Response(
        json.dumps({
            'similar_lines': similar_lines,
            'similar_paras': similar_paras,
            'wiki_results_filtered': wiki_results_filtered,
            'img_urls': img_urls
        }),
        mimetype='application/json',
        headers={
            'Cache-Control': 'no-cache',
            'Access-Control-Allow-Origin': '*'
        }
    )


app.run(
        host='0.0.0.0',
        port=11000,
        debug=False,
        threaded=True
    )
