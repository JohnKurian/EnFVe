from celery import Celery
import time
from celery.schedules import crontab
from celery.utils.log import get_task_logger

from allennlp.predictors.predictor import Predictor
from flask import Flask, Response, escape, request, render_template
import json

from pymongo import MongoClient

# from celery.task.control import revoke
# revoke(task_id, terminate=True)


import tweepy
import json

#Importing twitter credentials
consumer_key = "jjSz1RE4ftTNmqB1XuUTM22Fc"
consumer_secret = "5SNWrhQStzMp3UDwVY7YGuEofQ4QOBBP4rOo4hGnhKpdFQoVi9"
access_token = "68979886-IzdibLmYAx39y8PLNWA7kLPKl2rTDlLPCnf557I45"
access_token_secret = "tjVsF4mx4vS9JO0hPcS7b8qoP7oIZK1A8nX0aMwhkNEDG"

myclient = MongoClient()
nexus = myclient["nexus"]
tweets = nexus["tweets"]


# Accesing twitter from the App created in my account
def autorize_twitter_api():
    """
    This function gets the consumer key, consumer secret key, access token
    and access token secret given by the app created in your Twitter account
    and authenticate them with Tweepy.
    """
    # Get access and costumer key and tokens
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    return auth


class MyStreamListener(tweepy.StreamListener):
    """
    def on_status(self, status):
        print(status.text)
    """

    def __init__(self, filename, api=None):
        self.filename = filename

        tweepy.StreamListener.__init__(self, api=api)

    def on_data(self, raw_data):

        try:
            with open(self.filename, 'a') as file:
                file.write(raw_data)


        except Exception as e:
            print(e)

        raw_data = json.loads(raw_data)



        record = raw_data



        # x = tweets.update({'id': record['id']}, record, upsert=True)
        x = tweets.insert_one(record)

        print("Tweet colleted.")






logger = get_task_logger(__name__)

app = Celery('tasks', backend='amqp', broker='amqp://guest@localhost//')



#Creates the table for storing the tweets

#Connect to the streaming twitter API
api = tweepy.API(wait_on_rate_limit_notify=True,wait_on_rate_limit=True, auth_handler=autorize_twitter_api())




@app.task
def start_twitter_stream(term_to_search):
    terms = term_to_search
    streamer = tweepy.Stream(auth=autorize_twitter_api(),
                             listener=MyStreamListener(api=api, filename='tweets.txt'))
    streamer.filter(languages=["en"], track=terms)

# @app.task
# def reverse(string):
#     return string[::-1]

@app.task
def update_file():
    counter = 0
    while True:
        time.sleep(1)
        with open('sample.txt', 'w') as f:
            f.write(str(counter))
        logger.info(str(counter))
        counter += 1


@app.task
def check_status():
    time.sleep(20)
    return 'Job Completed'


@app.task
def add_two_numbers(x, y):
    print('here')
    return x + y


# @app.task
# def update_file():
#     return 'file updated'

@app.task
def save_to_file():
    counter = 0
    while True:
        counter += 1
        time.sleep(1)
        print('here')
        with open('output.txt', 'w') as f:
            f.write(str(counter))


app.conf.beat_schedule = {
    # Executes every minute.
    'execute-every-minute': {
        'task': 'tasks.update_file',
        'schedule': 1
    },
}
app.conf.timezone = 'Europe/London'


#rabbitmq-server

# celery -A tasks worker --loglevel=INFO





#Revoking a task by id

# from celery.task.control import revoke
# revoke(task_id, terminate=True)

#list all rabbit mq queues

#rabbitmqctl list_queues







#Remove all running tasks
#celery -A tasks purge


#remove all rabbitmq queues
# rabbitmqadmin -f tsv -q list queues name > q.txt
# while read -r name; do rabbitmqadmin -q delete queue name="${name}"; done < q.txt




