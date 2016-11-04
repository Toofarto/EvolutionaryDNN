from __future__ import absolute_import, division, print_function

import time, random, math
import numpy as np
import tensorflow as tf

from EEG_READ import *
from EEG_TF import *


# Build the graph
# VGG-16

time_start_build_network = time.time()

x = tf.placeholder(tf.float32, shape=[None, 40, 40, 1], name="x-input")
sz_y = 3
y_ = tf.placeholder(tf.float32, shape=[None, sz_y], name="y-input")
keep_prob = tf.placeholder(tf.float32)

network = conv_layer(x, 64, 3, tf.nn.relu)
network = conv_layer(network, 64, 3, tf.nn.relu)
network = max_pool_2x2(network)

network = conv_layer(network, 128, 3, tf.nn.relu)
network = conv_layer(network, 128, 3, tf.nn.relu)
network = max_pool_2x2(network)

network = conv_layer(network, 256, 3, tf.nn.relu)
network = conv_layer(network, 256, 3, tf.nn.relu)
network = conv_layer(network, 256, 3, tf.nn.relu)
network = max_pool_2x2(network)

network = conv_layer(network, 512, 3, tf.nn.relu)
network = conv_layer(network, 512, 3, tf.nn.relu)
network = conv_layer(network, 512, 3, tf.nn.relu)
network = max_pool_2x2(network)

network = conv_layer(network, 512, 2, tf.nn.relu)
network = conv_layer(network, 512, 2, tf.nn.relu)
network = conv_layer(network, 512, 2, tf.nn.relu)
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

print("Network Built")
print("Time for build network", time.time() - time_start_build_network)

time_start_preprocess = time.time()
pre_batch = '/data/klng/git/EvolutionaryDNN/Datasets/EEG_data/'
whole_data = np.array([unpickle(pre_batch + 's%.2d.dat'%i) for i in range(1, 32 + 1)])
whole_x = to_x(np.array(map(lambda x: x["data"], whole_data)))
whole_y = to_y(np.array(map(lambda x: x["labels"], whole_data)))
print(np.shape(whole_x))
print(np.shape(whole_y))

cut_off = 28 * 40 * 200
train_x = whole_x[:cut_off, :, :]
train_y = whole_y[:cut_off, :]
print(np.shape(train_x))
print(np.shape(train_y))
train_NP = NP_Dataset(train_x, train_y)

test_x = whole_x[cut_off:, :, :]
test_y = whole_y[cut_off:, :]
print(np.shape(test_x))
print(np.shape(test_y))
test_NP = NP_Dataset(test_x, test_y)

print("train and test data precess finished")
print("time for preprocess", time.time() - time_start_preprocess)

sess = tf.InteractiveSession()

sess.run(tf.initialize_all_variables())

start_train_time = time.time()
for i in range(300000):
    batch = train_NP.next_batch(32)
    if i%100 == 99:
        train_accuracy = accuracy.eval(feed_dict = {
            x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, epoch: %d, training accuracy %g"%(i+1, train_NP.get_epoch(), train_accuracy))
    else:  # Record a summary
        train_step.run(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 0.5})
print("Total Training Time:",(time.time() - start_train_time))
print("test accuracy %g"%accuracy.eval(feed_dict={
    x: test_x, y_:test_y, keep_prob: 1.0}))
