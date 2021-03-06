from __future__ import absolute_import, division, print_function

import time, random, math
import numpy as np

import tensorflow as tf

def weight_variable(shape, sz = 20.0):
    initial = tf.random_normal(shape, stddev=math.sqrt(2.0/sz))
    return tf.Variable(initial, name="weight")

def bias_variable(shape):
    # initial = tf.constant(0.0, shape=shape)
    initial = tf.zeros(shape)
    return tf.Variable(initial, name="bias")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def conv_layer(prev_layer, n_output, filter_width, activation = tf.identity, stride = 1, name = ""):
    name = name if name != "" else 'conv_' + str(random.randint(1, 1 << 30))
    
    width_prev_layer = int(prev_layer.get_shape()[3])
    with tf.name_scope(name):
        Weight_mat = weight_variable([filter_width, filter_width, width_prev_layer, n_output], 
                sz = filter_width * filter_width * width_prev_layer * n_output)
        res = tf.nn.conv2d(
            prev_layer, 
            Weight_mat, 
            strides=[1, stride, stride, 1], 
            padding='SAME') + bias_variable([n_output])
        return activation(res), tf.nn.l2_loss(Weight_mat)

def combine(t1, t2):
    sz_t1 = int(reduce(lambda x,y: x*y, t1.get_shape()[1:]))
    ft1 = tf.reshape(t1, [-1, sz_t1])
    sz_t2 = int(reduce(lambda x,y: x*y, t2.get_shape()[1:]))
    ft2 = tf.reshape(t2, [-1, sz_t2])
    return tf.concat(1, [ft1, ft2])

def lrelu(x, leak=0.2, name="lrelu"):
     with tf.variable_scope(name):
         f1 = 0.5 * (1 + leak)
         f2 = 0.5 * (1 - leak)
         return f1 * x + f2 * abs(x)

# TODO: Make it into a class
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
    sz_prev = int(reduce(lambda x,y: x*y, prev_layer.get_shape()[1:]))
    flat = tf.reshape(prev_layer, [-1, sz_prev])
    Weight = weight_variable([sz_prev, n_output], sz = n_output * sz_prev)
    Bias = bias_variable([n_output])
    return activation(tf.matmul(flat, Weight) + Bias, name = "activation"), tf.nn.l2_loss(Weight), tf.nn.l2_loss(Bias)

class FullConnectLayer(object):
    def __init__(self, name = ""):
        self.name = name if name != "" else 'fc_' + str(random.randint(1, 1 << 30))
        self.saved = False

    def update(self, weight, bias):
        self.saved_weight = weight
        self.saved_bias = bias
        self.saved = True

    def load(self, prev_layer, n_output, activation = tf.identity):
        with tf.variable_scope(self.name):
            sz_prev = int(reduce(lambda x,y: x*y, prev_layer.get_shape()[1:]))
            self.flat = tf.reshape(prev_layer, [-1, sz_prev])
            if not self.saved:
                self.Weight = weight_variable([sz_prev, n_output], sz = n_output)
                self.Bias = bias_variable([n_output])
            else:
                self.Weight = tf.Variable(self.saved_weight, name = "weight")
                self.Bias = tf.Variable(self.saved_bias, name = "bias")
            self.layer = activation(tf.matmul(self.flat, self.Weight) + self.Bias, name = "activation")

    def get(self):
        return (self.saved_weight, self.saved_bias)
