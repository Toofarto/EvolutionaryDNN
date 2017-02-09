# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import cPickle
import tflearn
import numpy as np
from tflearn.data_utils import to_categorical, pad_sequences
from tflearn.datasets import imdb

DATA_PATH = '../Datasets/EEG_data/'
NUM_DEFAULT_CHANNEL = 40

NUM_POS = 32
NUM_SPECTRUM = 5
NUM_LABEL = 4
NUM_POINT = 8064
NUM_SAMPLING = 128
NUM_VIDEO = 40
NUM_INTERVIEWEE = 32
NUM_OUTPUT_CLASS = 3
SIZE_WINDOW = 9
NUM_CHANNEL = NUM_POS * NUM_SPECTRUM
NUM_SPAN = NUM_POINT // NUM_SAMPLING
NUM_SPLITTED = NUM_SPAN // SIZE_WINDOW

NUM_TRAIN = 28
NUM_TEST = 4
SIZE_TRAIN = NUM_TRAIN * NUM_VIDEO * NUM_SPLITTED
SIZE_TEST = NUM_TEST * NUM_VIDEO * NUM_SPLITTED

with open(DATA_PATH + 'SplittedFlattenedPSD.dat', 'rb') as f:
    whole_data = cPickle.load(f)
wholeX, wholeY = whole_data['x'], whole_data['y']
trainX, trainY = wholeX[:SIZE_TRAIN , :, :], wholeY[:SIZE_TRAIN, :]
testX, testY = wholeX[SIZE_TRAIN:, :, :], wholeY[SIZE_TRAIN:, :]

# Network building
net = tflearn.input_data([None, SIZE_WINDOW, NUM_CHANNEL])
# net = tflearn.embedding(net, input_dim=10000, output_dim=128)
net = tflearn.lstm(net, 512, dropout=0.3)
net = tflearn.fully_connected(net, NUM_OUTPUT_CLASS, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.1,
                         loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True, batch_size=16)
unique, counts = np.unique(np.argmax(model.predict(testX), axis=1), return_counts=True)
print(dict(zip(unique, counts)))
