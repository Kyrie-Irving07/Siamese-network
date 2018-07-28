import tensorflow as tf
import numpy as np
import siamese
import visualize
from tensorflow.examples.tutorials.mnist import input_data

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

siamese = siamese.siamese()

train_step_all = tf.train.AdamOptimizer(0.005).minimize(siamese.loss, var_list=tf.trainable_variables())

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())
keep_prob = 1

embed0 = tf.convert_to_tensor(np.array([]))
y0 = tf.convert_to_tensor(np.array([]))
x0 = tf.convert_to_tensor(np.array([]))
x_test = mnist.test.images.reshape([-1, 28, 28])
y_test = mnist.test.labels

for i in range(40001):

    x1, y1 = mnist.train.next_batch(100)
    x2, y2 = mnist.train.next_batch(100)
    answer = (y1 == y2).astype(float)
    train_step = train_step_all
    _, loss, embed = sess.run([train_step, siamese.loss, siamese.out1], feed_dict={siamese.image1: x1,
                                                                                   siamese.image2: x2,
                                                                                   siamese.y_: answer})
    # print(tf.shape(x1).get_shape(), tf.shape(y1).get_shape(), tf.shape(embed).get_shape())
    '''
    if i != 0:
        embed0 = tf.concat([embed0, tf.reshape(embed, shape=[-1, 2])], 0)
        x0 = tf.concat([x0, tf.reshape(x1, shape=[-1, 784])], 0)
        y0 = tf.concat([y0, y1], 0)
    else:
        embed0 = tf.reshape(embed, shape=[-1, 2])
        y0 = y1
        x0 = tf.reshape(x1, shape=[-1, 784])

    if i >= 100:
        embed0 = embed0[100:]
        y0 = y0[100:]
        x0 = x0[100:]
    '''
    if i % 10 == 0:
        print("%d step, loss = %.3f" % (i, loss))

    if i % 2000 == 0:
        x1, y1 = mnist.test.next_batch(2000)
        embed = siamese.out1.eval(session=sess, feed_dict={siamese.image1: x1})
        x1 = x1.reshape([-1, 28, 28])
        visualize.visualize(embed, x1, y1, keep_prob)
