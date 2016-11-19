from __future__ import absolute_import, division, print_function

import time, random, math
import numpy as np

import tensorflow as tf

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

def weight_variable(shape, sz = 20.0):
    initial = tf.random_normal(shape, stddev=math.sqrt(2.0/sz))
    return tf.Variable(initial, name="weight")

def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial, name="bias")

def max_pool_2x2(x, name = ""):
    name = name if name != "" else 'max_pool_' + str(random.randint(1, 1 << 30))
    with tf.name_scope(name):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def conv_layer(prev_layer, n_output, filter_width, activation = tf.identity, stride = 1, name = ""):
    name = name if name != "" else 'conv_' + str(random.randint(1, 1 << 30))
    
    width_prev_layer = int(prev_layer.get_shape()[3])
    with tf.name_scope(name):
        Weight_mat = weight_variable([filter_width, filter_width, width_prev_layer, n_output], 
                sz = filter_width * filter_width * n_output)
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
        Weight = weight_variable([sz_prev, n_output], sz = n_output)
        Bias = bias_variable([n_output])
        return activation(tf.matmul(flat, Weight) + Bias, name = "activation")
