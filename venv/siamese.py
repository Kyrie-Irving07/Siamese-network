import tensorflow as tf
import numpy as np

class siamese:
    def __init__(self):
        self.image1 = tf.placeholder(dtype=tf.float32, shape=[None, 784], name="image1")
        self.image2 = tf.placeholder(dtype=tf.float32, shape=[None, 784], name="image2")
        # self.keep_prob = tf.placeholder(dtype=tf.float32)
        with tf.variable_scope("siamese") as scope:
            self.out1 = self.network(self.image1)
            scope.reuse_variables()
            self.out2 = self.network(self.image2)

        self.y_ = tf.placeholder(dtype=tf.float32, shape=[None])
        self.loss = self.loss_of_siamese()

    def accuracy(self):
        distance = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(self.out1, self.out2), 2), 1) + 1e-6)
        accu = tf.cast(tf.less(distance, 3), dtype=tf.float32)
        ans = tf.cast(tf.equal(self.y_, accu), dtype=tf.float32)
        accu = tf.reduce_mean(ans)
        return accu

    def network(self, x):
        x = tf.reshape(x, [-1, 28, 28, 1])
        net1 = self.conv(x, [5, 5, 1, 128], "net1")
        net1 = tf.nn.max_pool(net1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        net2 = self.conv(net1, [5, 5, 128, 64], "net2")
        net2 = tf.nn.max_pool(net2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        net3 = self.fully_con(tf.reshape(net2, shape=[-1, 49 * 64]), 2, name="net3")
        # net3 = self.senet(net2, 0.75)
        return net3
    
    def conv(self, prev, filter_size, name, padding='SAME'):
        initializer = tf.truncated_normal_initializer(stddev=0.1)
        w = tf.get_variable(name=name + 'w', shape=filter_size, initializer=initializer)
        b = tf.get_variable(name=name + 'b', initializer=tf.constant(0.1, shape=[filter_size[3]], dtype=tf.float32))
        # tf.add_to_collection(tf.GraphKeys.WEIGHTS, w)
        output = tf.nn.conv2d(prev, w, strides=[1, 1, 1, 1], padding=padding) + b
        return tf.nn.relu(output)
    
    def loss_of_siamese(self):
        distance = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(self.out1, self.out2), 2), 1) + 1e-6)
        margine = tf.constant(5, dtype=tf.float32)
        label = self.y_
        label_ = tf.subtract(1.0, label)
        term1 = tf.reduce_sum(tf.multiply(distance, label), 0)
        term2 = tf.reduce_sum(tf.multiply(tf.maximum(tf.subtract(margine, distance), 0), label_))
        term1 = tf.pow(term1, 2) + term1
        term2 = tf.pow(term2, 2) + term2
        # regularizer = tf.contrib.layers.l2_regularizer(scale=1.0/100)
        # reg_term = tf.contrib.layers.apply_regularization(regularizer)
        loss_of_siamese = term1 + term2  # + reg_term
        return loss_of_siamese

    def senet(self, net, rate):
        shape = net.get_shape().as_list()
        fsq = tf.nn.avg_pool(net, ksize=[1, shape[1], shape[2], 1], strides=[1, 1, 1, 1], padding='SAME')
        fsq = self.fully_con(fsq, int(shape[3] * rate), name='l1', activation=tf.nn.relu)
        fex = self.fully_con(fsq, int(shape[3]), name='l2', activation=tf.nn.sigmoid)
        net = tf.multiply(net, fex)
        net = tf.reshape(net, [-1, shape[1] * shape[2] * shape[3]])
        w = tf.get_variable('senet_w', [net.get_shape().as_list()[1], 2],
                            initializer=tf.truncated_normal_initializer(stddev=0.1))
        b = tf.get_variable('senet_b', initializer=tf.constant(0.1, dtype=tf.float32, shape=[2]))
        net = tf.matmul(net, w) + b
        return net

    def fully_con(self, net, outdim, name, activation=None, use_bias=True):
        with tf.variable_scope(name + 'fully_con'):
            return tf.layers.dense(net, units=outdim, activation=activation, use_bias=use_bias)
