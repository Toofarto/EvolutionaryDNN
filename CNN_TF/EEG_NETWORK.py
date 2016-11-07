from __future__ import absolute_import, division, print_function

import time, random, math
import numpy as np
import tensorflow as tf

from EEG_READ import *
from EEG_TF import *


# Build the graph
# VGG-16

time_start_preprocess = time.time()

x = tf.placeholder(tf.float32, shape=[None, 40, 40, 1], name="x-input")
sz_y = 3
y_ = tf.placeholder(tf.float32, shape=[None, sz_y], name="y-input")
keep_prob = tf.placeholder(tf.float32)

network = conv_layer(x, 64, 3, tf.nn.relu)
network = tf.nn.dropout(network, keep_prob)
network = conv_layer(network, 64, 3, tf.nn.relu)
network = max_pool_2x2(network)

network = conv_layer(network, 128, 3, tf.nn.relu)
network = tf.nn.dropout(network, keep_prob)
network = conv_layer(network, 128, 3, tf.nn.relu)
network = max_pool_2x2(network)

network = conv_layer(network, 256, 3, tf.nn.relu)
network = tf.nn.dropout(network, keep_prob)
network = conv_layer(network, 256, 3, tf.nn.relu)
network = tf.nn.dropout(network, keep_prob)
network = conv_layer(network, 256, 3, tf.nn.relu)
network = max_pool_2x2(network)

network = conv_layer(network, 512, 3, tf.nn.relu)
network = tf.nn.dropout(network, keep_prob)
network = conv_layer(network, 512, 3, tf.nn.relu)
network = tf.nn.dropout(network, keep_prob)
network = conv_layer(network, 512, 3, tf.nn.relu)
network = max_pool_2x2(network)

network = full_connect_layer(network, 512, tf.nn.relu)
network = tf.nn.dropout(network, keep_prob)
network = full_connect_layer(network, 512, tf.nn.relu)
network = tf.nn.dropout(network, keep_prob)
network = full_connect_layer(network, sz_y, tf.nn.softmax)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(network), reduction_indices=[1]))
train_step = tf.train.MomentumOptimizer(5e-6, 0.9).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(network, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print("Network (VGG-16) was Built")
