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
p_keep_input = 0.8
p_keep_hidden = 0.9

W_1 = tf.Variable(tf.random_normal(shape=[n_visible, n_hidden_1], stddev=0.01), name='W_1')
W_2 = tf.Variable(tf.random_normal(shape=[n_hidden_1, n_hidden_2], stddev=0.01), name='W_2')
b_1 = tf.Variable(tf.zeros([n_hidden_1]), name='b_1')
b_2 = tf.Variable(tf.zeros([n_hidden_2]), name='b_2')

W_final_init = 1 / n_hidden_2
W_final = tf.Variable(tf.random_uniform([n_hidden_2, 10], minval=-W_final_init, maxval=W_final_init))

def model(X, W_1, b_1, W_2, b_2, W_final, p_keep_input, p_keep_hidden):
	X = tf.nn.dropout(X, p_keep_input)
	h1 = tf.nn.relu(tf.matmul(X, W_1) + b_1)
	h1 = tf.nn.dropout(h1, p_keep_hidden)
	h2 = tf.nn.relu(tf.matmul(h1, W_2) + b_2)
	h2 = tf.nn.dropout(h2, p_keep_hidden)
	return  tf.matmul(h2, W_final)

X = tf.placeholder("float", [None, 784])
Y = tf.placeholder("float", [None, 10])
py_x = model(X, W_1, b_1, W_2, b_2, W_final, p_keep_input, p_keep_hidden)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y))
train_op = tf.train.GradientDescentOptimizer(0.02).minimize(cost)
predict_op = tf.argmax(py_x, 1)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

for i in range(100):
    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
        sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
    print i, np.mean(np.argmax(teY, axis=1) ==
                     sess.run(predict_op, feed_dict={X: teX, Y: teY}))

