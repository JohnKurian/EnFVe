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

import pandas as pd
#handling data
import pandas as pd
import numpy as np
from scipy import stats
from operator import itemgetter



#handling information
import re
import json

#handling plots
import matplotlib.pyplot as plt
import seaborn as sns

#for network creation
import networkx as nx
import tweepy
import math
from collections import Counter

from pymongo import MongoClient

myclient = MongoClient()
nexus = myclient["nexus"]
tweets = nexus["tweets"]
users = nexus["users"]
reports = nexus["reports"]


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


@app.task
def update_report():
    #     logger.info(str('updated report task called.'))

    myclient = MongoClient()
    nexus = myclient["nexus"]
    tweets = nexus["tweets"]
    users = nexus["users"]
    reports = nexus["reports"]

    users = nexus["users"].find({})
    queries = []
    for user in users:
        print(user['sub'])
        queries.append({'$match': {"user.sub": user['sub']}})
    reps = nexus.reports.aggregate(queries)

    hashtags_list = []
    for r in reps:
        hashtags_list.append(r['hashtags'])

    unique_data = [list(x) for x in set(tuple(sorted(x)) for x in hashtags_list)]

    print('unique hashtag list:', unique_data)

    for idx, hashtags in enumerate(unique_data):
        print('fetching report for:', unique_data[idx])
        queries = []
        for hashtag in hashtags:
            if hashtag[0] == '#':
                hashtag = hashtag[1:]
                queries.append({"entities.hashtags.text": hashtag})
        if len(queries) == 0:
            continue
        query = [{'$match': {'$or': queries}}]
        tweets = nexus.tweets.aggregate(query)

        tweet_list = []
        for tweet in tweets:
            tweet_list.append(tweet)

        tweets_df = pd.DataFrame(tweet_list)

        all_hashtags = []
        for e in tweets_df.entities:
            hashtags = [t['text'] for t in e['hashtags']]
            all_hashtags = all_hashtags + hashtags

        from collections import Counter
        c = Counter(all_hashtags)
        most_common_tuples = c.most_common()
        sorted_keys = sorted(c, key=c.get, reverse=True)

        hashtag_dict = {}
        for t in most_common_tuples:
            hashtag_dict[t[0]] = t[1]

        hashtag_wordclouds = []
        for t in most_common_tuples:
            hashtag_wordclouds.append({'text': t[0], 'value': t[1]})

        for t in most_common_tuples:
            tweets_df['#' + t[0]] = False

        for i, e in enumerate(tweets_df.entities):
            hashtags = [t['text'] for t in e['hashtags']]
            for hashtag in hashtags:
                tweets_df.at[i, '#' + hashtag] = True

        def apply_func(x):
            if not isinstance(x, float):
                if 'full_text' in x:
                    return x['full_text']
                else:
                    return float('nan')
            else:
                return float('nan')

        tweets_df['full_text'] = tweets_df['extended_tweet'].apply(lambda x: apply_func(x))
        tweets_df.full_text.fillna(tweets_df.text, inplace=True)

        import re
        from collections import Counter
        import requests

        import extraction
        import requests

        import favicon

        url_list = []
        for text in tweets_df['text'].values.tolist():
            urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
            url_list += urls

        url_list = list(filter(lambda url: len(url) > 13, url_list))

        c = Counter(url_list)
        most_common_urls = c.most_common()
        sorted_urls = sorted(c, key=c.get, reverse=True)

        news_articles = []
        twitter_domain_url = 'https://twitter.com'

        url_list = []
        for e in tweets_df.entities:
            for url_obj in e['urls']:
                if url_obj['expanded_url'][:19] != 'https://twitter.com':
                    url_list.append(url_obj['expanded_url'])

        c = Counter(url_list)
        most_common_urls = c.most_common()
        sorted_urls = sorted(c, key=c.get, reverse=True)

        for i, url in enumerate(sorted_urls):
            print('count value:', i)
            if i == 20:
                break
            try:
                news_article_dict = {}

                html = requests.get(url).text
                extracted = extraction.Extractor().extract(html, source_url=url)
                icon_url = favicon.get(url)[0][0]

                if url[:19] != twitter_domain_url:
                    #                     print('title:', extracted.title)
                    #                     print('description:', extracted.description)
                    #                     print(extracted.image, url, icon_url)

                    news_article_dict['title'] = extracted.title
                    news_article_dict['description'] = extracted.description
                    news_article_dict['favicon'] = icon_url
                    news_article_dict['image'] = extracted.image
                    news_article_dict['url'] = url
                    news_article_dict['share_count'] = most_common_urls[i][1]

                    news_articles.append(news_article_dict)

            except:
                print(url)

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

        api = tweepy.API(wait_on_rate_limit_notify=True, wait_on_rate_limit=True, auth_handler=autorize_twitter_api())

        retweeted_ids = []
        for rt in tweets_df['retweeted_status'].values:
            if not isinstance(rt, float):
                retweeted_ids.append(rt['id'])

        c = Counter(retweeted_ids)
        most_common_tuples = c.most_common()
        sorted_keys = sorted(c, key=c.get, reverse=True)
        most_common_tuples

        ids = []
        for t in most_common_tuples:
            ids.append(t[0])
        ids = ids[:20]

        statuses = api.statuses_lookup(ids)

        statuses_list = []
        for status in statuses:
            t_obj = {
                'text': status._json['text'],
                'id': status._json['id'],
                'tweet_link': 'https://twitter.com/i/web/status/' + status._json['id_str'],
                'user_screen_name': status._json['user']['screen_name'],
                'json': status._json
            }

            t = (t_obj, status._json['retweet_count'])
            statuses_list.append(t)

        s = sorted(statuses_list, key=lambda x: x[1])
        #         s.reverse()

        viral_tweets = [x[0] for x in s]

        # Create a second dataframe to put important information
        tweets_final = pd.DataFrame(
            columns=["created_at", "id", "in_reply_to_screen_name", "in_reply_to_status_id", "in_reply_to_user_id",
                     "retweeted_id", "retweeted_screen_name", "user_mentions_screen_name", "user_mentions_id",
                     "text", "user_id", "screen_name", "followers_count"])

        # Columns that are going to be the same
        equal_columns = ["created_at", "id", "text"]
        tweets_final[equal_columns] = tweets_df[equal_columns]

        # Get the basic information about user
        def get_basics(tweets_final):
            print(tweets_df["user"])
            tweets_final["screen_name"] = tweets_df["user"].apply(lambda x: x["screen_name"])
            tweets_final["user_id"] = tweets_df["user"].apply(lambda x: x["id"])
            tweets_final["followers_count"] = tweets_df["user"].apply(lambda x: x["followers_count"])
            return tweets_final

        # Get the user mentions
        def get_usermentions(tweets_final):
            # Inside the tag 'entities' will find 'user mentions' and will get 'screen name' and 'id'
            tweets_final["user_mentions_screen_name"] = tweets_df["entities"].apply(
                lambda x: x["user_mentions"][0]["screen_name"] if x["user_mentions"] else np.nan)
            tweets_final["user_mentions_id"] = tweets_df["entities"].apply(
                lambda x: x["user_mentions"][0]["id_str"] if x["user_mentions"] else np.nan)
            return tweets_final

        # Get retweets
        def get_retweets(tweets_final):
            # Inside the tag 'retweeted_status' will find 'user' and will get 'screen name' and 'id'
            tweets_final["retweeted_screen_name"] = tweets_df["retweeted_status"].apply(
                lambda x: x["user"]["screen_name"] if x is not np.nan else np.nan)
            tweets_final["retweeted_id"] = tweets_df["retweeted_status"].apply(
                lambda x: x["user"]["id_str"] if x is not np.nan else np.nan)
            return tweets_final

        # Get the information about replies
        def get_in_reply(tweets_final):
            # Just copy the 'in_reply' columns to the new dataframe
            tweets_final["in_reply_to_screen_name"] = tweets_df["in_reply_to_screen_name"]
            tweets_final["in_reply_to_status_id"] = tweets_df["in_reply_to_status_id"]
            tweets_final["in_reply_to_user_id"] = tweets_df["in_reply_to_user_id"]
            return tweets_final

        # Lastly fill the new dataframe with the important information
        def fill_df(tweets_final):
            get_basics(tweets_final)
            get_usermentions(tweets_final)
            get_retweets(tweets_final)
            get_in_reply(tweets_final)
            return tweets_final

        # Get the interactions between the different users
        def get_interactions(row):
            # From every row of the original dataframe
            # First we obtain the 'user_id' and 'screen_name'
            user = row["user_id"], row["screen_name"]
            # Be careful if there is no user id
            if user[0] is None:
                return (None, None), []

            # The interactions are going to be a set of tuples
            interactions = set()

            # Add all interactions
            # First, we add the interactions corresponding to replies adding the id and screen_name
            interactions.add((row["in_reply_to_user_id"], row["in_reply_to_screen_name"]))
            # After that, we add the interactions with retweets
            interactions.add((row["retweeted_id"], row["retweeted_screen_name"]))
            # And later, the interactions with user mentions
            interactions.add((row["user_mentions_id"], row["user_mentions_screen_name"]))

            # Discard if user id is in interactions
            interactions.discard((row["user_id"], row["screen_name"]))
            # Discard all not existing values
            interactions.discard((None, None))
            # Return user and interactions
            return user, interactions

        tweets_final = fill_df(tweets_final)

        tweets_final = tweets_final.where((pd.notnull(tweets_final)), None)

        graph = nx.Graph()

        for index, tweet in tweets_final.iterrows():
            user, interactions = get_interactions(tweet)
            user_id, user_name = user
            tweet_id = int(tweet["id"])
            # tweet_sent = tweet["sentiment"]
            for interaction in interactions:
                int_id, int_name = interaction
                graph.add_edge(user_id, int_id, tweet_id=tweet_id)

                graph.node[user_id]["name"] = user_name
                graph.node[user_id]["text"] = tweet['text']
                graph.node[int_id]["name"] = int_name
                graph.node[int_id]["text"] = tweet['text']

        degrees = [val for (node, val) in graph.degree()]

        largest_subgraph = max(nx.connected_component_subgraphs(graph), key=len)

        graph_centrality = nx.degree_centrality(largest_subgraph)

        max_de = max(graph_centrality.items(), key=itemgetter(1))

        graph_closeness = nx.closeness_centrality(largest_subgraph)

        max_clo = max(graph_closeness.items(), key=itemgetter(1))

        graph_betweenness = nx.betweenness_centrality(largest_subgraph, normalized=True, endpoints=False)

        max_bet = max(graph_betweenness.items(), key=itemgetter(1))

        all_bet = sorted(graph_betweenness.items(), key=itemgetter(1))

        all_bet.reverse()

        all_de = sorted(graph_centrality.items(), key=itemgetter(1))
        all_de.reverse()

        ids = []
        for de in all_de:
            #     print(graph.node[de[0]])
            ids.append(de[0])
        #     print(de[0])
        #     print(graph.node[de[0]]['name'])

        ids = ids[:10]

        users = api.lookup_users(user_ids=ids)

        user_list = []
        for user in users:
            user_list.append(user._json)

        ##############################################
        influencers = user_list
        ##############################################

        #         print(hashtag_wordclouds)
        #         print(news_articles)
        #         print(viral_tweets)
        #         print(influencers)

        hashtags = unique_data[idx]

        hashtags_len = len(hashtags)
        #         print('hashtags:', hashtags)
        #         hashtags = ['#'+hashtag for hashtag in hashtags]

        rprts = nexus.reports.find({"hashtags": {"$size": hashtags_len, "$all": hashtags}})

        for r in rprts:
            report_id = r['id']
            print('report_id:', r['id'])
            print('report:', r)

            r['hashtag_wordclouds'] = hashtag_wordclouds
            r['news_articles'] = news_articles
            r['viral_tweets'] = viral_tweets
            r['influencers'] = influencers

            print('updating reports...')
            nexus.reports.update({'id': report_id}, r, upsert=True)


app.conf.beat_schedule = {
    # Executes every minute.
    'execute-every-2-minutes': {
        'task': 'tasks.update_report',
        'schedule': 120
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




