from celery import Celery
import time
from celery.schedules import crontab
from celery.utils.log import get_task_logger
import requests

from allennlp.predictors.predictor import Predictor
from flask import Flask, Response, escape, request, render_template
import json

from celery.task.control import revoke
from pymongo import MongoClient

myclient = MongoClient()
nexus = myclient["nexus"]
tweets = nexus["tweets"]
users = nexus["users"]
reports = nexus["reports"]



import tweepy
import json

from tasks import start_twitter_stream

import twint

app = Flask(__name__)
logger = get_task_logger(__name__)




@app.route('/stop_stream', methods=['GET', 'POST'])
def stop():
    print('stopping stream..')

    try:
        with open('jobs.txt', 'r') as f:
            jobs = [x.rstrip('\n') for x in f.readlines()]
            for job_id in jobs:
                revoke(job_id, terminate=True)
    except:
        pass

    return {}


@app.route('/get_tweets', methods=['GET', 'POST'])
def get_tweets():
    hashtags = request.json['hashtags']
    queries = []
    for hashtag in hashtags:
        # queries.append({'entities.hashtags': {'$elemMatch': {'text': hashtag}}})
        # queries.append({ "$match": { "entities.hashtags.text": hashtag } })
        # queries.append({"entities.hashtags.text": hashtag})
        # '$match': { '$or': [{"entities.hashtags.text": 'dave'}, {"entities.hashtags.text": 'john'}]}

        queries.append({"entities.hashtags.text": hashtag})


    query = [{'$match': {'$or': queries}}]
    # tweets = nexus.tweets.find({'entities.hashtags': {'$elemMatch': {'text': hashtags[0]}}})
    tweets = nexus.tweets.aggregate(query)
    print(tweets)
    tweet_list = []
    for tweet in tweets:
        tweet_list.append(tweet)

    print(len(tweet_list))
    return 'fetched'


@app.route('/update_user', methods=['GET', 'POST'])
def update_user():
    user = request.json['user']
    x = nexus.users.update({'sub': user['sub']}, user, upsert=True)
    print('user updated')

    return Response(
        json.dumps({'result': True}),
        mimetype='application/json',
        headers={
            'Cache-Control': 'no-cache',
            'Access-Control-Allow-Origin': '*'
        }
    )


@app.route('/get_report', methods=['GET', 'POST'])
def get_report():
    print(request.json)
    report_id = request.json['report_id']
    print(request)
    report = nexus.reports.find_one({'id': report_id})
    del report['_id']

    return Response(
        json.dumps({'report': report}),
        mimetype='application/json',
        headers={
            'Cache-Control': 'no-cache',
            'Access-Control-Allow-Origin': '*'
        }
    )


@app.route('/get_reports', methods=['GET', 'POST'])
def get_reports_for_user():
    user = request.json['user']
    # report = request.json['report']

    report_list = []
    reports  = nexus.reports.find({'user.sub': user['sub']})

    for report in reports:
        print(report)
        temp_report = {}
        temp_report['id'] = report['id']
        temp_report['hashtags'] = report['hashtags']
        report_list.append(temp_report)

    return Response(
        json.dumps({'reports': report_list}),
        mimetype='application/json',
        headers={
            'Cache-Control': 'no-cache',
            'Access-Control-Allow-Origin': '*'
        }
    )








@app.route('/', methods=['GET', 'POST'])
def answer():

    print('removing old jobs..')

    user = request.json['user']

    try:
        with open('jobs.txt', 'r') as f:
            jobs = [x.rstrip('\n') for x in f.readlines()]
            for job_id in jobs:
                revoke(job_id, terminate=True)
    except:
        pass



    print('getting new list of hashtags..')
    user_hashtags = request.json['hashtags']
    hashtags = request.json['hashtags']
    with open('hashtags.txt', 'a') as f:
        for hashtag in hashtags:
            f.write(hashtag + '\n')

    with open('hashtags.txt', 'r') as f:
        hashtags = [x.rstrip('\n') for x in f.readlines()]

    hashtags = list(set(hashtags))

    print('starting twitter stream..')

    stream_job = start_twitter_stream.delay(hashtags)

    report_obj = {
        'id': stream_job.id,
        'user': user,
        'hashtags': user_hashtags
    }

    id = stream_job.id

    x = nexus.reports.update({'id': id}, report_obj, upsert=True)

    with open('jobs.txt', 'a') as f:
        f.write(str(stream_job.id) + '\n')
    logger.info(stream_job)



    return Response(
                json.dumps({'job_id': stream_job.id}),
                mimetype='application/json',
                headers={
                    'Cache-Control': 'no-cache',
                    'Access-Control-Allow-Origin': '*'
                }
            )



app.run(
        host='0.0.0.0',
        port=16000,
        debug=False,
        threaded=True
    )


# r = requests.post('http://0.0.0.0:16000/', json={'hashtags': ['#insurance']})

#stopping stream
# r = requests.post('http://0.0.0.0:16000/stop_stream')

#get tweets for hashtags
# r = requests.post('http://0.0.0.0:16000/get_tweets', json={'hashtags': ['#insurance']})


# db.sample.aggregate([
#     // Filter possible documents
#     { "$match": { "filtermetric.class": "s2" } },
#
#     // Unwind the array to denormalize
#     { "$unwind": "$filtermetric" },
#
#     // Match specific array elements
#     { "$match": { "filtermetric.class": "s2" } },
#
#     // Group back to array form
#     { "$group": {
#         "_id": "$_id",
#         "filtermetric": { "$push": "$filtermetric" }
#     }}
# ])



