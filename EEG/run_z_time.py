# input the path of EEG dataset on commandline
from __future__ import absolute_import, division, print_function

import time, random, math, sys
import numpy as np
import tensorflow as tf

from tf import *
from network_z_time import *
from data_z_time import *

time_start_preprocess = time.time()
#pre_batch = '/data/klng/git/EvolutionaryDNN/Datasets/EEG_data/'
pre_batch = sys.argv[1]
whole_data = np.array([unpickle(pre_batch + 's%.2d.dat'%i) for i in range(1, 32 + 1)])
whole_x = to_x(np.array(map(lambda x: x["data"], whole_data)))
whole_y = to_y(np.array(map(lambda x: x["labels"], whole_data)))

cut_off = 28 * 40 * 200
train_x = whole_x[:cut_off, :, :]
train_y = whole_y[:cut_off, :]
train_NP = NP_Dataset(train_x, train_y)

test_x = whole_x[cut_off:, :, :]
test_y = whole_y[cut_off:, :]
test_NP = NP_Dataset(test_x, test_y)

print("train and test data precess finished")
print("time for preprocess", time.time() - time_start_preprocess)

sess = tf.InteractiveSession()

sess.run(tf.initialize_all_variables())

start_train_time = time.time()
last_epoch = -1
for i in range(300000):
    batch = train_NP.next_batch(64)
    if i%500 == 499:
        train_accuracy = accuracy.eval(feed_dict = {
            x: batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, epoch: %d, training accuracy %g"%(i+1, train_NP.get_epoch(), train_accuracy))
    elif train_NP.get_epoch() > last_epoch:
        last_epoch = train_NP.get_epoch()
        test_batch = test_NP.next_batch(512)
        print("finished epoch: %d"%last_epoch)
        print("test accuracy %g"%accuracy.eval(feed_dict={x: test_batch[0], y_: test_batch[1], keep_prob: 1.0}))
    else:  # Record a summary
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
print("Total Training Time:",(time.time() - start_train_time))
test_time = 5
final_test_accuracy = 0.0
for i in range(test_time):
    test_batch = test_NP.next_batch(512)
    final_test_accuracy += accuracy.eval(feed_dict={x: test_batch[0], y_: test_batch[1], keep_prob: 1.0}) / test_time
print("final_test_accuracy = %f"%final_test_accuracy)
