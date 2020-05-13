import nltk
import os
from google.cloud import bigquery
from google.cloud.bigquery.client import Client
from annoy import AnnoyIndex
import spacy
import neuralcoref
from datetime import datetime
import subprocess
import redis
import json
import tensorflow as tf
import tensorflow_hub as hub

# nltk.download('punkt')
# nltk.download('stopwords')

nlp = spacy.load('en')
neuralcoref.add_to_pipe(nlp)
print('getting USE model...')
embed = hub.load('https://tfhub.dev/google/universal-sentence-encoder-large/5')
print('fetched model.')

r = redis.Redis(
    host='127.0.0.1',
    port=6379)

D = 512
NUM_TREES = 10
ann = AnnoyIndex(D, metric='angular')

try:
    # ann.load('article.index')
    print('annoy loaded successfully.')
except:
    pass

# #start time
# with open('time.txt', 'r') as f:
#  s = datetime.fromtimestamp(int(f.read()))
#  start_object = s.strftime('%Y-%m-%d %H:%M:%S')
# #print(start_object)


# # current date and time
# now = datetime.now()
# timestamp = int(datetime.timestamp(now))
# with open("time.txt", "w") as f:
#     f.write(str(timestamp))
# c = datetime.fromtimestamp(timestamp)
# current_object = c.strftime('%Y-%m-%d %H:%M:%S')
# #print(c)
# print(current_object)


os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'api_key.json'
bq_client = Client()
query = "select full_content,url, publishedAt from `jadgs-262219.newsapi.news_articles`"
# query = "select url, full_content from `jadgs-262219.newsapi.news_articles` where publishedAt >= '" + start_object + "' and publishedAt < '" + current_object + "'"
query_job = bq_client.query(query)
results = query_job.result()

line_counter = 0
embedding_counter = 0

for row in results:
    if row[0] is not None:
        text = row[0]
        url = row[1]
        publishedAt = datetime.timestamp(row[2])
        doc = nlp(text)
        clusters = doc._.coref_clusters
        # print("clusters ",clusters)
        # print ("\n\n")
        resolved_coref = doc._.coref_resolved
        print("Resolved by NeuralCoref: \n")  # NeuralCoref
        # print(resolved_coref)
        sentences = nltk.sent_tokenize(resolved_coref)  # sentance tokenize
        lines = [x.replace('\n', '').replace('\\', '') for x in sentences]

        if len(lines) > 0:
            for line in lines:
                payload = {'url': url, 'line': line, 'vector_index': str(line_counter), 'publishedAt': publishedAt}
                r.set(str(line_counter), json.dumps({'url': url, 'line': line, 'publishedAt': publishedAt}))
                line_counter += 1

            print('building embeddings..')
            embeddings = embed(lines)
            # print(lines)

            for index, e in enumerate(embeddings):
                # print(embedding_counter, line_counter)
                ann.add_item(embedding_counter, e)
                embedding_counter += 1
                # with open("embeddings.txt", "a") as txt_file:
                #   txt_file.write(str(e) + "\n")

ann.build(NUM_TREES)
ann.save('article_full.index')


# subprocess.Popen(['gsutil', 'cp', 'article_100.index', 'gs://embeddings_newsarticles/article_100.index']). # pushing into Bucket
