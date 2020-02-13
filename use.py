# import tensorflow as tf
# import tensorflow_hub as hub
#
#
# print('loading use..')
# module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
# model = hub.load(module_url)
# print('loading finished.')
#
# def generate_embeddings(messages_in):
#     # generate embeddings
#     with tf.compat.v1.Session() as tf.compat.v1.Session:
#         tf.compat.v1.Session.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.tables_initializer()])
#         message_embeddings = tf.compat.v1.Session.run(model(messages_in))
#
#     return message_embeddings
#
#
# # let's create some test sentanaces
# test_messages = [
#     "AquilaDB is a Resillient, Replicated, Decentralized, Host neutral storage for Feature Vectors along with Document Metadata.",
#     "Do k-NN retrieval from anywhere, even from the darkest rifts of Aquila (in progress). It is easy to setup and scales as the universe expands.",
#     "AquilaDB is a Resillient, Replicated, Decentralized, Host neutral storage for Feature Vectors along with Document Metadata.",
#     "Do k-NN retrieval from anywhere, even from the darkest rifts of Aquila (in progress). It is easy to setup and scales as the universe expands.",
#     "AquilaDB is a Resillient, Replicated, Decentralized, Host neutral storage for Feature Vectors along with Document Metadata.",
#     "Do k-NN retrieval from anywhere, even from the darkest rifts of Aquila (in progress). It is easy to setup and scales as the universe expands."]
#
#
# test_messages = test_messages + test_messages + test_messages + test_messages + test_messages + test_messages + test_messages + test_messages + test_messages + test_messages + test_messages + test_messages
#
# test_messages = ['one sentence']
#
# import time
# session = tf.compat.v1.Session()
# session.run(tf.compat.v1.global_variables_initializer())
# session.run(tf.tables_initializer())
# tf.executing_eagerly()
#
# print('generating embeddings..')
# start_time = time.time()
# # print(generate_embeddings(test_messages))
#
#
# embeddings = model(test_messages)
#
# e = session.run(embeddings)
#
# print(e)
#
#
#
# print('time:', time.time() - start_time)
# print('done.')

#Function so that one session can be called multiple times.
#Useful while multiple calls need to be done for embedding.
import tensorflow as tf
import tensorflow_hub as hub
def embed_useT(module):
    with tf.Graph().as_default():
        sentences = tf.placeholder(tf.string)
        embed = hub.load(module)
        embeddings = embed(sentences)
        session = tf.train.MonitoredSession()
    return lambda x: session.run(embeddings, {sentences: x})
tf.executing_eagerly()
embed_fn = embed_useT('https://tfhub.dev/google/universal-sentence-encoder/4')
messages = [
    "we are sorry for the inconvenience",
    "we are sorry for the delay",
    "we regret for your inconvenience",
    "we don't deliver to baner region in pune",
    "we will get you the best possible rate"
]
start_time = time.time()
print('first call..')
embed_fn(messages)
print('second call..')
embed_fn(messages)