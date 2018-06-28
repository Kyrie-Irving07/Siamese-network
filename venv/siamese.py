import tensorflow as tf

class siamese:
    def __init__(self):
        self.image1 = tf.placeholder(dtype=tf.float32, shape=[None, 784], name="image1")
        self.image2 = tf.placeholder(dtype=tf.float32, shape=[None, 784], name="image2")

        with tf.name_scope("siamese") as scope:
            self.out1=self.network(self.image1)
            scope.reuse_variables()
            self.out2=self.network(self.iamge2)

        self.y_=tf.placeholder(dtype=tf.float32,shape=[None])
        self.loss=self.loss_of_siamese()

    def network(self,x):
        net1=self.layer(x,1024,"net1")
        net1=tf.nn.relu(net1)
        net2=self.layer(net1,1024,"net2")
        net2=tf.nn.relu(net2)
        net3=self.layer(net2,2,"net3")