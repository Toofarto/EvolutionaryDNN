
# coding: utf-8

# In[1]:

from __future__ import absolute_import, division, print_function

import time, random
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf
sess = tf.InteractiveSession()

def variable_summaries(var, name):
    """Attach a lot of summaries to a Tensor."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.scalar_summary('mean/' + name, mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.scalar_summary('stddev/' + name, stddev)
        tf.scalar_summary('max/' + name, tf.reduce_max(var))
        tf.scalar_summary('min/' + name, tf.reduce_min(var))
        tf.histogram_summary(name, var)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name="weight")

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name="bias")

def max_pool_2x2(x, name = ""):
    name = name if name != "" else 'max_pool_' + str(random.randint(1, 1 << 30))
    with tf.name_scope(name):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def conv_layer(prev_layer, n_output, filter_width, activation = tf.identity, stride = 1, name = ""):
    name = name if name != "" else 'conv_' + str(random.randint(1, 1 << 30))
    
    width_prev_layer = int(prev_layer.get_shape()[3])
    with tf.name_scope(name):
        Weight_mat = weight_variable([filter_width, filter_width, width_prev_layer, n_output])
        variable_summaries(Weight_mat, name+'/weights')
        res = tf.nn.conv2d(
            prev_layer, 
            Weight_mat, 
            strides=[1, stride, stride, 1], 
            padding='SAME') + bias_variable([n_output])
        return activation(res)

def full_connect_layer(prev_layer, n_output, activation = tf.identity, name = ""):
    """
    transform the previous layer to a "flat" matrix
    And then fully connect to the next layer

    input:
    prev_layer: The previous layer
    activation: ghd activation function of the nervous
    K: size of the output layer 

    ouput:
    a "flat" matrix with size K
    """
    name = name if name != "" else 'fc_' + str(random.randint(1, 1 << 30))
    with tf.name_scope(name):
        sz_prev = int(reduce(lambda x,y: x*y, prev_layer.get_shape()[1:]))
        flat = tf.reshape(prev_layer, [-1, sz_prev])
        Weight = weight_variable([sz_prev, n_output])
        Bias = bias_variable([n_output])
        return activation(tf.matmul(flat, Weight) + Bias, name = "activation")


# In[50]:

x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name="x-input")
sz_y = 10
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
network = max_pool_2x2(network)

network = conv_layer(network, 512, 3, tf.nn.relu)
network = conv_layer(network, 512, 3, tf.nn.relu)
network = max_pool_2x2(network)

network = full_connect_layer(network, 4096, tf.nn.relu)
network = tf.nn.dropout(network, keep_prob)
network = full_connect_layer(network, 4096, tf.nn.relu)
network = tf.nn.dropout(network, keep_prob)
network = full_connect_layer(network, sz_y, tf.nn.softmax)

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(network), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-3).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(network, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[ ]:

import numpy as np

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

class NP_Dataset(object):
    def __init__(self, pX, pY):
        self._X = pX
        #self._Y = [map(lambda x: x == y, range(17)) for y in pY]
        self._Y = pY
        assert np.shape(self._X)[0] == np.shape(self._Y)[0]
        self._n_sample = np.shape(self._X)[0]
        self._index_in_epoch = 0
        self._epoch_completed = 0
        
    def next_batch(self, batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._n_sample:
            assert batch_size <= self._n_sample
            self._epoch_completed += 1
            # Shuffle
            perm = np.arange(self._n_sample) 
            np.random.shuffle(perm)
            self._X = self._X[perm]
            self._Y = self._Y[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
        end = self._index_in_epoch
        return self._X[start:end], self._Y[start:end]

def to_X(X, n_data):
    X.resize(n_data, 3, 32, 32)
    X = X.swapaxes(1, 3)
    return X
    
def to_Y(Y, n_data):
    Y = np.array(Y)
    Y.resize(n_data)
    res = np.zeros((n_data, sz_y))
    res[np.arange(n_data), Y] = 1
    return res

pre_batch = '../Datasets//cifar-10-batches-py/'
cifar_batch = [unpickle('../Datasets//cifar-10-batches-py/data_batch_%d'%i) for i in range(1, 6)]
train_x = np.array(map(lambda x: x['data'], cifar_batch))
train_x = to_X(train_x, 50000)
train_y = np.array(map(lambda x: x['labels'], cifar_batch))
train_y = to_Y(train_y, 50000)
cifar_train = NP_Dataset(train_x, train_y)

test = unpickle(pre_batch + 'test_batch') 
test_x = to_X(test['data'], 10000)
test_y = to_Y(test['labels'], 10000)
cifar_test = NP_Dataset(test_x, test_y)


# In[ ]:

sess.run(tf.initialize_all_variables())

start_train_time = time.time()
for i in range(2000):
    batch = cifar_train.next_batch(256)
    if i%100 == 99:
        train_accuracy = accuracy.eval(feed_dict = {
            x:batch[0], y_: batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    else:  # Record a summary
        train_step.run(feed_dict={
                x: batch[0], y_: batch[1], keep_prob: 0.5})
end_train_time = time.time()
print("Total Training Time:",(end_train_time - start_train_time))

