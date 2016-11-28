# input the path of EEG dataset on commandline
from __future__ import absolute_import, division, print_function

import time, random, math, sys
import sklearn as sk
from sklearn import metrics
import numpy as np
import tensorflow as tf

from tf import *
from network_psd import *
from data_psd import *

time_start_preprocess = time.time()
#pre_batch = '/data/klng/git/EvolutionaryDNN/Datasets/EEG_data/'
pre_batch = sys.argv[1]
n_GPU = int(sys.argv[2])

if n_GPU > 0:
    for i in range(n_GPU):
        with tf.device("/gpu:%d"%i):
            build_network()
else :
    build_network()
from network_psd import *
print("PSD, Network (VGG-9) was Built")

whole_data = np.array([unpickle(pre_batch + 's%.2d.dat'%i) for i in range(1, 32 + 1)])
whole_x = to_x(np.array(map(lambda x: x["data"], whole_data)))
whole_y = to_y(np.array(map(lambda x: x["labels"], whole_data)))

cut_off = 28 * 40 * 63
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
        test_batch = test_NP.next_batch(1024)
        print("finished epoch: %d"%last_epoch)
        val_accuracy, y_pred = sess.run([accuracy, y_p], feed_dict={x: test_batch[0], y_: test_batch[1], keep_prob: 1.0})
        print("test accuracy %g"%val_accuracy)
        y_true = np.argmax(test_batch[1], 1)
	# print("Precision", metrics.precision_score(y_true, y_pred))
	# print("Recall", metrics.recall_score(y_true, y_pred))
	# print("f1_score", metrics.f1_score(y_true, y_pred))
	print("confusion_matrix")
	print(metrics.confusion_matrix(y_true, y_pred))
    else:  # Record a summary
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
print("Total Training Time:",(time.time() - start_train_time))
test_time = 5
final_test_accuracy = 0.0
for i in range(test_time):
    test_batch = test_NP.next_batch(512)
    final_test_accuracy += accuracy.eval(feed_dict={x: test_batch[0], y_: test_batch[1], keep_prob: 1.0}) / test_time
print("final_test_accuracy = %f"%final_test_accuracy)
