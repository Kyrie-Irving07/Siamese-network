import tensorflow as tf

class siamese:
    def __init__(self):
        self.image1 = tf.placeholder(dtype=tf.float32, shape=[None, 784], name="image1")
        self.image2 = tf.placeholder(dtype=tf.float32, shape=[None, 784], name="image2")

        with tf.variable_scope("siamese") as scope:
            self.out1 = self.network(self.image1)
            scope.reuse_variables()
            self.out2 = self.network(self.image2)

        self.y_ = tf.placeholder(dtype=tf.float32, shape=[None])
        self.loss = self.loss_of_siamese()

    def network(self, x):
        net1 = self.layer(x, 1024, "net1")
        net1 = tf.nn.relu(net1)
        net2 = self.layer(net1, 1024, "net2")
        net2 = tf.nn.relu(net2)
        net3 = self.layer(net2, 2, "net3")
        return net3
    
    def layer(self,prev,next,name):
        initializer = tf.truncated_normal_initializer(stddev=0.1)
        w = tf.get_variable(name=name + 'w', shape=[prev.get_shape()[1], next], initializer=initializer)
        b = tf.get_variable(name=name + 'b', initializer=tf.constant(0.1, shape=[next], dtype=tf.float32))
        output = tf.nn.bias_add(tf.matmul(prev, w), b)
        return output
    
    def loss_of_siamese(self):
        distance = tf.sqrt(tf.reduce_sum(tf.pow(tf.subtract(self.out1, self.out2), 2), 1) + 1e-6)
        margine = tf.constant(5, dtype=tf.float32)
        label = self.y_
        label_ = tf.subtract(1.0, label)
        term1 = tf.reduce_sum(tf.multiply(distance, label), 0)
        term2 = tf.reduce_sum(tf.multiply(tf.maximum(tf.subtract(margine, distance), 0), label_))
        loss_of_siamese = term1 + term2
        return loss_of_siamese