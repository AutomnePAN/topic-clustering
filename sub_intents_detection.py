#!/usr/bin/env python
# coding: utf-8

# # Sub Intents Study
# ------


from __future__ import absolute_import, division, print_function, unicode_literals
import warnings

warnings.filterwarnings('ignore')
import pandas as pd
import requests
import numpy as np
import keras
import tensorflow as tf
import tensorflow_hub as hub
import os
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import GlobalAveragePooling1D
from keras.layers import Reshape
from keras.layers import Dropout
from keras.layers.merge import add
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Lambda, Input, Flatten
from keras import Model
from sklearn import preprocessing
from sklearn.metrics.pairwise import cosine_similarity

tf.keras.backend.clear_session()  # For easy reset of notebook state.

from topic_clustering_models import Modeling


class DataPreprocessing(object):
    def __init__(self):
        self.url_data = "https://github.com/google-research-datasets/Taskmaster/raw/master/TM-1-2019/self-dialogs.json"
        self.data = requests.get(self.url_data).json()
        self.url_elmo = "https://tfhub.dev/google/elmo/2"
        self.ELMO = hub.Module(self.url_elmo)
        self.model_elmo = None
        self.intent2phrase = {}

    def ELMOO(self, x):
        return self.ELMO(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]

    def build_model_elmo(self):
        input_text1 = Input(shape=(None,), dtype=tf.string)
        embeding1 = Lambda(self.ELMOO, output_shape=(1, 1024))(input_text1)
        model = Model(inputs=input_text1, outputs=embeding1)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model_elmo = model

    def phrase_embedding(self, phrase):
        X = np.array([[phrase], [phrase]])
        #     print(X)
        if not self.model_elmo:
            self.build_model_elmo()
        res = self.model_elmo.predict(X)
        return res[0]

    def intents_clustering(self):
        intents2phrases = {}
        for i in range(len(self.data)):
            for ut in self.data[i]['utterances']:
                if ut['speaker'] == 'USER':
                    if 'segments' in ut.keys():
                        for seg in ut['segments']:
                            if 'annotations' in seg.keys():
                                for anno in seg['annotations']:
                                    name = anno['name']
                                    if name in intents2phrases.keys():
                                        intents2phrases[name].append(ut['text'])
                                    else:
                                        intents2phrases[name] = [ut['text']]

        intents = np.array(list(intents2phrases.keys()))
        for intent in intents:
            # print("Paddling: intent: ", intent)
            df = pd.DataFrame(np.array(intents2phrases[intent]))
            df.to_csv("Sub_topics_strings/%s.csv" % intent)
        self.intent2phrase = intents2phrases
        return intents2phrases


class SubIntentsModeling(object):
    ''' In this class , we use the notations below'''
    ''' topic means: the grand theme of the conversation, e.g.: "restaurant" '''
    ''' subtopic means: the small theme belonging to the topic, e.g. "restaurant.time" '''
    ''' intents means: the intent of the user, which kind of express the attitude of user, e.g. "restaurant.time.accept" '''

    def __init__(self, intent2phrase):
        self.intent2phrase = intent2phrase
        self.intents = intent2phrase.keys()
        self.topic = ['auto', 'coffee', 'movie', 'pizza', 'restaurant', 'uber']
        self.topic2Intents = None

        #       the sub_topic dictionary stores the subtopics of each topic and their coding
        self.sub_topic = {'restaurant': {"time": 1, "location": 2, "num": 3, "name": 4, "type": 5, "others": 6},
                          'auto': {"name": 1, "year": 2, "reason": 3, "date": 4, "others": 5},
                          'movie': {"time": 1, "location": 2, "num": 3, "name": 4, "type": 5, "price": 6, "others": 7},
                          'coffee': {"size": 1, "location": 2, "num": 3, "name": 4, "type": 5, "preference": 6,
                                     "others": 7},
                          'pizza': {"size": 1, "location": 2, "num": 3, "name": 4, "type": 5, "others": 6},
                          'uber': {"time": 1, "location": 2, "num": 3, "duration": 4, "type": 5, "others": 6}}

        #       Two dictionarys to stock the training set according to topics
        self.train_X = {}
        self.train_Y = {}
        for t in self.topic:
            self.train_X[t] = None
            self.train_Y[t] = None

        #       Models pretrained
        self.url_data = "https://github.com/google-research-datasets/Taskmaster/raw/master/TM-1-2019/self-dialogs.json"
        self.data = requests.get(self.url_data).json()
        self.url_elmo = "https://tfhub.dev/google/elmo/2"
        self.ELMO = hub.Module(self.url_elmo)

        #       One dictionary to store the models of prediction according to different topics
        self.models = {}
        for t in self.topic:
            if os.path.exists("subtopic_clustering_model/%s.h5" % t):
                self.build_model_lstm(t)
                self.models[t].load_weights("subtopic_clustering_model/%s.h5" % t)
            else:
                self.models[t] = None

    def trained(self):
        for t in self.topic:
            if self.models[t] is None:
                return False
        return True;

    def ELMOO(self, x):
        return self.ELMO(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]

    def topic2Intents_clustering(self):
        topic2Intents = {}
        for t in self.topic:
            topic2Intents[t] = []
        for i in self.intents:
            for t in self.topic:
                if i.find(t) > -1:
                    topic2Intents[t].append(i)
                    break
        self.topic2Intents = topic2Intents
        return

    def transform_array(self, Y, numofSubclass):  # transforms Y from class num array to appropriate format
        new_Y = []
        for i in range(len(Y)):
            toadd = [0] * numofSubclass
            toadd[int(float(Y[i][0])) - 1] = 1
            new_Y.append(toadd)

        return np.array(new_Y)

    def get_training_set(self, topic):
        if topic not in self.topic:
            print("Error: wrong topic!")
            return -1

        #       If the intents are not yet classified according to topic
        if not self.topic2Intents:
            self.topic2Intents_clustering()
        intents_topic = self.topic2Intents[topic]

        #       Get the subtopics and classify the intents according to the subtopics
        sub_topic = self.sub_topic[topic]
        intent2Subtopic = {}
        for i in intents_topic:
            for st in sub_topic.keys():
                if i.find(st) > -1:
                    intent2Subtopic[i] = sub_topic[st]
                    break
                intent2Subtopic[i] = sub_topic["others"]

        train_X = np.array([], dtype=np.int64).reshape(0, 1)
        train_Y = np.array([], dtype=np.int64).reshape(0, 1)

        for intent in intents_topic:
            #           Get the phrases
            X_ = np.array(self.intent2phrase[intent])
            X_ = X_.reshape(len(X_), 1)
            train_X = np.vstack([train_X, X_])
            Y_ = intent2Subtopic[intent] * np.ones(X_.shape[0]).reshape([X_.shape[0], 1])
            train_Y = np.vstack([train_Y, Y_])

        #         shuffle the train set
        train_set = np.concatenate((train_Y, train_X), axis=1)
        np.random.shuffle(train_set)
        train_Y = train_set[:, :1]
        train_X = train_set[:, 1:]
        train_Y = self.transform_array(train_Y, len(self.sub_topic[topic].keys()))

        self.train_X[topic] = train_X
        self.train_Y[topic] = train_Y

        return train_X, train_Y

    def build_model_lstm(self, topic):

        numofSubclass = len(self.sub_topic[topic].keys())
        input_text1 = Input(shape=(None,), dtype=tf.string)
        embeding1 = Lambda(self.ELMOO, output_shape=(1, 1024))(input_text1)
        reshape = Reshape((1, 1024), input_shape=(1, 1024))(embeding1)
        ltsm = LSTM(128, return_sequences=False, input_shape=(1, 1024))(reshape)
        out = Dense(numofSubclass, activation="softmax")(ltsm)
        model = Model(inputs=input_text1, outputs=out)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.models[topic] = model

        return model

    def training_models(self, topic, batch_size=32, epochs=4):
        if self.models[topic] == None:
            self.build_model_lstm(topic)
        if self.train_X[topic] == None or self.train_Y[topic] == None:
            self.get_training_set(topic)

        model = self.models[topic]
        train_X = self.train_X[topic]
        train_Y = self.train_Y[topic]
        model.fit(train_X, train_Y, epochs=epochs, batch_size=batch_size, shuffle=True, validation_split=0.2)
        self.models[topic] = model
        model.save_weights("subtopic_clustering_model/%s.h5" % topic)
        print("{} model saved".format(topic));
        return

    def int2Subtopic(self, i, topic):
        for k in self.sub_topic[topic].keys():
            if self.sub_topic[topic][k] == i:
                return k
        return "Error"

    def prediction(self, topic, phrases):
        """phrases have to be a numpy array of strings"""

        if len(phrases) == 1:
            X = np.array([phrases[0], phrases[0]])
            prediction = self.models[topic].predict(X)
            res = [self.int2Subtopic(i + 1, topic) for i in np.argmax(prediction, axis=1)]
            # print("Result of prediction: ")
            # print(64 * "=")
            # for i in range(1):
                # print(res[i], end=":")
                # print(X[i][0])
            return res[0]
        else:
            X = phrases
            prediction = self.models[topic].predict(X)
            res = [self.int2Subtopic(i + 1, topic) for i in np.argmax(prediction, axis=1)]

            # for i in range(len(res)):
            #     print(res[i], end=":")
            #     print(X[i][0])
            return res


if __name__ == "__main__":
    dp = DataPreprocessing()
    i2phr = dp.intents_clustering()
    SIM = SubIntentsModeling(i2phr)
    if not SIM.trained():
        SIM.training_models("restaurant")
        SIM.training_models("movie")
        SIM.training_models("auto")
        SIM.training_models("pizza")
        SIM.training_models("uber")
        SIM.training_models("coffee")
    test_phrases = np.array([["Where is the nearest restaurant?"], ["This Evening"], ["I want this"],
                  ["We have totally 4"], ["My dad, mom, me and my sister"], ["Do you have some recommends"],
                  ["Thank you! Bye"]])
    res = SIM.prediction("restaurant", test_phrases)
    print("Result of prediction: ")
    print(64 * "=")
    for i in range(len(res)):
        print("The predicted intent of \"{}\" is {}".format(test_phrases[i][0], res[i]))
