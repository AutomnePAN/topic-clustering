# -*- coding: utf-8 -*-
# Built by Riade Benbaki and Haolin Pan
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import requests
import numpy as np
import keras
import tensorflow as tf
import tensorflow_hub as hub
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import GlobalAveragePooling1D
from keras.layers import Reshape
from keras.layers import Dropout
from keras.layers.merge import add
import os
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, Lambda, Input, Flatten
from keras import Model


class Modeling(object):

    def __init__(self):
        self.url_data = "https://github.com/google-research-datasets/Taskmaster/raw/master/TM-1-2019/self-dialogs.json"
        self.data = requests.get(self.url_data).json()

        self.numofclasses = 7
        self.url_elmo = "https://tfhub.dev/google/elmo/2"
        self.ELMO = hub.Module(self.url_elmo)
        self.batchsize = 32
        self.model_built = False
        self.classes_arr = ['auto', 'coffee', 'movie', 'non-opening', 'pizza', 'restaurant', 'uber']
        self.X = None
        self.Y = None
        return

    def transform_array(self, Y):  # transforms Y from class name array to appropriate format
        classes_arr = np.unique(Y)
        classes_dict = dict()
        for i, class_n in enumerate(np.unique(Y)):
            classes_dict[class_n] = i
        # print(classes_dict)
        for i in range(len(Y)):
            toadd = [0] * self.numofclasses
            toadd[classes_dict[Y[i]]] = 1
            Y[i] = list(toadd);
        return np.array(Y)

    def dataPreprocessing(self):
        X, Y = [], []
        X_val, Y_val = [], []
        samplestotakefromeachclass = 1000
        dd = {'auto': 0,
              'coffee': 0,
              'movie': 0,
              'pizza': 0,
              'restaurant': 0,
              'uber': 0,
              'non-opening': 0}

        for ut in self.data:
            sent = ut["utterances"][0]
            class_to_add = ut["instruction_id"].split("-")[0]
            if dd[class_to_add] < samplestotakefromeachclass:
                X.append(sent["text"])
                Y.append(class_to_add)
                dd[class_to_add] += 1
            else:
                X_val.append(sent["text"])
                Y_val.append(class_to_add)

            class_to_add = 'non-opening'
            if dd[class_to_add] < samplestotakefromeachclass:
                X.append(ut["utterances"][np.random.randint(1, len(ut["utterances"]))]["text"])
                Y.append(class_to_add)
                dd[class_to_add] += 1
            else:
                X_val.append(sent["text"])
                Y_val.append(class_to_add)

        self.X = np.array(X)
        self.Y = self.transform_array(Y)

        self.X_val = np.array(X_val)[:6000]
        self.Y_val = self.transform_array(Y_val)[:6000]
        return X, Y, X_val, Y_val;

    def ELMOO(self, x):
        return self.ELMO(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["default"]

    def build_model_bider(self):
        input_text = Input(shape=(1,), dtype=tf.string)
        embedding = Lambda(self.ELMOO, output_shape=(1, 1024))(input_text)
        # print(embeding1.shape)
        reshape = Reshape((1, 1024), input_shape=(1, 1024))(embedding)
        x = Bidirectional(
            LSTM(units=128, return_sequences=True, recurrent_dropout=0.2, dropout=0.2, input_shape=(1, 1024)))(reshape)
        x_rnn = Bidirectional(LSTM(units=128, return_sequences=False, recurrent_dropout=0.2, dropout=0.2))(x)
        out = Dense(self.numofclasses, activation="softmax")(x_rnn)
        model = Model(inputs=input_text, outputs=out)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def build_model_lstm(self):
        input_text1 = Input(shape=(None,), dtype=tf.string)
        embeding1 = Lambda(self.ELMOO, output_shape=(1, 1024))(input_text1)
        reshape = Reshape((1, 1024), input_shape=(1, 1024))(embeding1)
        ltsm = LSTM(128, return_sequences=False, input_shape=(1, 1024))(reshape)
        out = Dense(self.numofclasses, activation="softmax")(ltsm)
        model = Model(inputs=input_text1, outputs=out)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def build_model_cnn_1d(self):
        input_text = Input(shape=(None,), dtype=tf.string)
        embedding = Lambda(self.ELMOO, output_shape=(1, 1024))(input_text)
        reshape = Reshape((1024 // 8, 8), input_shape=(1, 1024))(embedding)
        cnn1 = Conv1D(1024 // 8, 8, activation='relu')(reshape)
        cnn2 = Conv1D(1024 // 8, 8, activation='relu')(cnn1)
        mp = MaxPooling1D(3, 1)(cnn2)
        cnn3 = Conv1D(128, 8, activation='relu')(mp)
        cnn4 = Conv1D(128, 8, activation='relu')(cnn3)
        dp = Dropout(0.5)(cnn4)
        gap = GlobalAveragePooling1D()(dp)
        output = Dense(self.numofclasses, activation="softmax")(gap)
        model = Model(inputs=input_text, output=output)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def modelBuildingAndTraining(self):
        if self.X is None or self.Y is None:
            self.dataPreprocessing()

        self.model_lstm = self.build_model_lstm()
        if os.path.exists("topic_clustering_model/lstm.h5"):
            self.model_lstm.load_weights("topic_clustering_model/lstm.h5")
        else:
            self.model_lstm.fit(self.X, self.Y, epochs=4, batch_size=self.batchsize, validation_split=0.2)
            self.model_lstm.save_weights("topic_clustering_model/lstm.h5")

        self.model_bidir = self.build_model_bider()
        if os.path.exists("topic_clustering_model/bidir.h5"):
            self.model_bidir.load_weights("topic_clustering_model/bidir.h5")
        else:
            self.model_bidir.fit(self.X, self.Y, epochs=10, batch_size=self.batchsize, validation_split=0.2)
            self.model_bidir.save_weights("topic_clustering_model/bidir.h5")

        self.model_cnn1d = self.build_model_lstm()
        if os.path.exists("topic_clustering_model/cnn1d.h5"):
            self.model_cnn1d.load_weights("topic_clustering_model/cnn1d.h5")
        else:
            self.model_cnn1d.fit(self.X, self.Y, epochs=8, batch_size=self.batchsize, validation_split=0.2)
            self.model_cnn1d.save_weights("topic_clustering_model/cnn1d.h5")

    # def modelTraining(self, x):
    #     if self.model_built:
    #         if x == 0:
    #             print("LSTM")
    #             self.model_lstm.fit(self.X, self.Y, epochs=3, batch_size=self.batchsize, validation_split=0.2)
    #             self.model_lstm.save_weights("/topic_clustering_model/lstm.h5")
    #         if x == 1:
    #             print("BIDIR")
    #             self.model_bidir.fit(self.X, self.Y, epochs=4, batch_size=self.batchsize, validation_split=0.2)
    #             self.model_bidir.save_weights("/topic_clustering_model/bidre.h5")
    #         if x == 2:
    #             print("CNN1D")
    #             self.model_cnn1d.fit(self.X, self.Y, epochs=8, batch_size=self.batchsize, validation_split=0.2)
    #             self.model_cnn1d.save_weights("/topic_clustering_model/cnn1d.h5")
    #         return
    #     else:
    #         print("Error: Models not built yet")
    #         return

    def prediction(self, phrs):
        if len(phrs) == 1:
            X = np.array([ phrs[0], phrs[0]])
        else:
            X = np.array(phrs)
        prediction = self.model_lstm.predict(X)
        a = [self.classes_arr[i] for i in np.argmax(prediction, axis=1)]
        return a




if __name__ == "__main__":
    s = Modeling()
    s.dataPreprocessing()
    s.modelBuildingAndTraining()
    print("Test the model of topic clustering: ")
    print("Here we try to detect an opening phrase of one scenario, one conversation; ")
    print("Normally, this would be a phrase to express a need")
    print("Otherwise if the phrase has no information about the scenario, for expample, \" Ok, that's it \" we regard this as  \"non-opnening\"")
    print(64 * "=")
    test_phrases = [["Where is the nearest Starbucks ?"],["i need to repair my car"],\
                    ["I need a ride from home"],["I want to order something to eat"],["can you activate"],\
                    ["I want a table in center city"], ["Ok that's it!"]]
    res = s.prediction(test_phrases)
    for i in range(len(res)):
        print("The predicted topic of \"{} \" is : {}".format(test_phrases[i][0], res[i]))
    print(64 * "-")
