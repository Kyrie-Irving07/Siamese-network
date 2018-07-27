import tensorflow as tf
import siamese
import visualize
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=False)

siamese = siamese.siamese()

train_step_all = tf.train.AdamOptimizer(0.005).minimize(siamese.loss, var_list=tf.trainable_variables())

sess = tf.Session()
sess.run(tf.global_variables_initializer())

keep_prob = 1

for i in range(40001):
    x1, y1 = mnist.train.next_batch(100)
    x2, y2 = mnist.train.next_batch(100)
    # answer = tf.cast(tf.equal(y1, y2), dtype=tf.float32)
    answer = (y1 == y2).astype(float)
    train_step = train_step_all
    _, loss = sess.run([train_step, siamese.loss], feed_dict={siamese.image1: x1,
                                                              siamese.image2: x2,
                                                              siamese.y_: answer})
    if i % 10 == 0:
        print("%d step, loss = %.3f" % (i, loss))
    if i % 10000 == 0:
        embed = siamese.out1.eval(session=sess, feed_dict={siamese.image1: mnist.test.images})
        x_test = mnist.test.images.reshape([-1, 28, 28])
        y_test = mnist.test.labels
        visualize.visualize(embed, x_test, y_test, keep_prob)
