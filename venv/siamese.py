import tensorflow as tf
import numpy as np

class siamese:
    def __init__(self):
        self.image1 = tf.placeholder(dtype=tf.float32, shape=[None, 784], name="image1")
        self.image2 = tf.placeholder(dtype=tf.float32, shape=[None, 784], name="image2")
        self.keep_prob = tf.placeholder(dtype=tf.float32)
        with tf.variable_scope("siamese") as scope:
            self.out1 = self.network(self.image1)
            scope.reuse_variables()
            self.out2 = self.network(self.image2)

        self.y_ = tf.placeholder(dtype=tf.float32, shape=[None])
        self.loss = self.loss_of_siamese()

    def network(self, x):
        x = tf.reshape(x, [-1, 28, 28, 1])
        net1 = self.layer(x, [5, 5, 1, 64], "net1")
        net1 = tf.nn.max_pool(net1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        net1 = tf.nn.relu(net1)
        net2 = self.layer(net1, [5, 5, 64, 32], "net2")
        net2 = tf.nn.max_pool(net2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        net2 = tf.nn.relu(net2)
        net3 = self.senet(net2, 1.5)
        return net3
    
    def layer(self, prev, filter_size, name):
        initializer = tf.truncated_normal_initializer(stddev=0.1)
        w = tf.get_variable(name=name + 'w', shape=filter_size, initializer=initializer)
        b = tf.get_variable(name=name + 'b', initializer=tf.constant(0.1, shape=[filter_size[3]], dtype=tf.float32))
        #tf.add_to_collection(tf.GraphKeys.WEIGHTS, w)
        output = tf.nn.conv2d(prev, w, [1, 1, 1, 1], padding='SAME') + b
        return output
    
    def loss_of_siamese(self):
        distance = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(self.out1, self.out2), 2), 1) + 1e-6)
        margine = tf.constant(5, dtype=tf.float32)
        label = self.y_
        label_ = tf.subtract(1.0, label)
        term1 = tf.reduce_sum(tf.multiply(distance, label), 0)
        term2 = tf.reduce_sum(tf.multiply(tf.maximum(tf.subtract(margine, distance), 0), label_))
        term1 = tf.pow(term1, 2) + term1
        term2 = tf.pow(term2, 2) + term2
        regularizer = tf.contrib.layers.l2_regularizer(scale=1.0/100)
        reg_term = tf.contrib.layers.apply_regularization(regularizer)
        loss_of_siamese = term1 + term2 + reg_term
        return loss_of_siamese

    def senet(self, net, rate):
        Fsq = tf.nn.avg_pool(net, ksize=[1, 7, 7, 1], strides=[1, 1, 1, 1], padding='SAME')
        Fsq = tf.reshape(Fsq, [-1, 32])
        w1 = tf.get_variable('w1', shape=[32, int(32 / rate)], initializer=tf.truncated_normal_initializer(stddev=0.1))
        b1 = tf.get_variable('b1', initializer=tf.constant(0.1, dtype=tf.float32, shape=[int(32 / rate)]))
        Fsq = tf.matmul(Fsq, w1) + b1
        Fsq = tf.nn.relu(Fsq)
        w2 = tf.get_variable('w2', shape=[int(32 / rate), 32], initializer=tf.truncated_normal_initializer(stddev=0.1))
        b2 = tf.get_variable('b2', initializer=tf.constant(0.1, dtype=tf.float32, shape=[32]))
        Fsq = tf.matmul(Fsq, w2) + b2
        output = tf.reshape(net, [-1, 7*7, 32]) * tf.reshape(Fsq, [-1, 49, 32])
        output = tf.reshape(output, [-1, 7*7*32])
        w = tf.get_variable('w', shape=[7*7*32, 2], initializer=tf.truncated_normal_initializer(stddev=0.1))
        b = tf.get_variable('b', initializer=tf.constant(0.1, tf.float32, shape=[2]))
        output = tf.matmul(output, w) + b
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, w)
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, w1)
        tf.add_to_collection(tf.GraphKeys.WEIGHTS, w2)
        return output
