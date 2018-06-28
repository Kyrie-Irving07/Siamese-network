import tensorflow as tf
import siamese
import visualize
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot = False)

siamese = siamese.siamese()
train_step = tf.train.AdamOptimizer(0.005).minimize(siamese.loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(20000):
    x1,y1 = mnist.train.next_batch(100)
    x2,y2 = mnist.train.next_batch(100)
    #answer = tf.cast(tf.equal(y1, y2), dtype=tf.float32)
    answer = (y1 == y2).astype(float)
    _,loss = sess.run([train_step, siamese.loss], feed_dict={siamese.image1:x1, siamese.image2:x2, siamese.y_:answer})
    if i % 10 == 0:
        print("%d step, loss = %.3f" %(i, loss))
    if i % 5000 == 0 and i > 0:
        embed = siamese.out1.eval(session=sess, feed_dict={siamese.image1: mnist.test.images})
        x_test = mnist.test.images.reshape([-1, 28, 28])
        y_test = mnist.test.labels
        visualize.visualize(embed, x_test, y_test)