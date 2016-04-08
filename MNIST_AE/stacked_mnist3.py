import numpy as np
import input_data
import tensorflow as tf
import math
import matplotlib
import matplotlib.pyplot as plt

mnist_width = 28
n_visible = mnist_width * mnist_width
n_hidden_1 = 450
n_hidden_2 = 100
corruption_levels = [0.7, 0.8]

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

def plot_mnist_digit(image1, image2):
    """ Plot a single MNIST image."""
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1)
    image = np.reshape(image1, (28, 28))
    ax.matshow(image, cmap = matplotlib.cm.binary)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    ax = fig.add_subplot(1, 2, 2)
    image = np.reshape(image2, (28, 28))
    ax.matshow(image, cmap = matplotlib.cm.binary)
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.show()

X = tf.placeholder("float", [None, n_visible], name='X')

W_1 = tf.Variable(tf.random_normal(shape=[n_visible, n_hidden_1], stddev=0.01), name='W_1')
b_1 = tf.Variable(tf.zeros([n_hidden_1]), name='b_1')
W_1_prime = tf.transpose(W_1)
b_1_prime = tf.Variable(tf.zeros([n_visible]), name='b_1_prime')

def model1(X, W_1, b_1, W_1_prime, b_1_prime):
    X = tf.nn.dropout(X, corruption_levels[0])
    Y = tf.nn.sigmoid(tf.matmul(X, W_1) + b_1)
    Z = tf.nn.sigmoid(tf.matmul(Y, W_1_prime) + b_1_prime)
    return Z

Z1 = model1(X, W_1, b_1, W_1_prime, b_1_prime)
cost1 = tf.reduce_sum(tf.pow(X - Z1, 2))
train_op1 = tf.train.RMSPropOptimizer(0.001, 0.8).minimize(cost1)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

print "\n\nstarting layer 1 training"
diff = sess.run(Z1, feed_dict={X: teX}) - teX
print "original error is: " + str(np.sum(diff * diff))
for i in range(30):
    for start, end in zip(range(1, len(trX), 100), range(100, len(trX), 100)):
        sess.run(train_op1, feed_dict={X: trX[start:end]})
    print i, sess.run(cost1, feed_dict={X: teX})

w_1_val_original = sess.run(W_1)
print "Layer 1 done!\n\n\n\n"
# myy = sess.run(Z1, feed_dict={X: teX})
# plot_mnist_digit(teX[0], myy[0])
###############################################################################
W_1_copy = tf.Variable(tf.random_normal(shape=[n_visible, n_hidden_1], stddev=0.01), trainable=False, name='W_1_copy_copy')
b_1_copy = tf.Variable(tf.zeros([n_hidden_1]), trainable=False, name='b_1_copy')
W_1_prime_copy = tf.transpose(W_1_copy)
b_1_prime_copy = tf.Variable(tf.zeros([n_visible]), trainable=False, name='b_1_prime_copy')

W_2 = tf.Variable(tf.random_normal(shape=[n_hidden_1, n_hidden_2], stddev=0.01), name='W_2')
b_2 = tf.Variable(tf.zeros([n_hidden_2]), name='b_2')
W_2_prime = tf.transpose(W_2)
b_2_prime = tf.Variable(tf.zeros([n_hidden_1]), name='b_2_prime')

def model2(X, W_2, b_2, W_2_prime, b_2_prime):
    X = tf.nn.dropout(X, corruption_levels[1])
    Y = tf.nn.sigmoid(tf.matmul(X, W_2) + b_2)
    Z = tf.nn.sigmoid(tf.matmul(Y, W_2_prime) + b_2_prime)
    return Z

latent1 = tf.nn.sigmoid(tf.matmul(X, W_1_copy) + b_1_copy)
Z2 = model2(latent1, W_2, b_2, W_2_prime, b_2_prime)
cost2 = tf.reduce_sum(tf.pow(latent1 - Z2, 2))
train_op2 = tf.train.GradientDescentOptimizer(0.02).minimize(cost2)

init = tf.initialize_variables([W_2, b_2, b_2_prime, W_1_copy, b_1_copy, b_1_prime_copy], name="init2")
sess.run(init)
sess.run(W_1_copy.assign(W_1))
sess.run(b_1_copy.assign(b_1))
sess.run(b_1_prime_copy.assign(b_1_prime))

print "\n\nstarting layer 2 training"
diff = sess.run(Z2, feed_dict={X: teX}) - sess.run(latent1, feed_dict={X: teX})
print "original error is: " + str(np.sum(diff * diff))
for i in range(20):
    for start, end in zip(range(0, len(trX), 500), range(500, len(trX), 500)):
        sess.run(train_op2, feed_dict={X: trX[start:end]})
    print i, sess.run(cost2, feed_dict={X: teX})

# make sure that weights did not change! =)
print "Layer 2 done!\n\n\n\n"
print np.sum(sess.run(W_1) - w_1_val_original)
print np.sum(sess.run(W_1_copy) - w_1_val_original)
# myy2 = sess.run(tf.nn.sigmoid(tf.matmul(Z2, W_1_prime) + b_1_prime), feed_dict={X: teX})
# plot_mnist_digit(teX[0], myy2[0])


###############################################################################
saver = tf.train.Saver()
save_path = saver.save(sess, "stackedweights.sess")
