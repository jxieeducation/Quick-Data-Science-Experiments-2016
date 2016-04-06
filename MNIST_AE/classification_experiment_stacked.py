import numpy as np
import input_data
import tensorflow as tf
import math
import matplotlib
import matplotlib.pyplot as plt

mnist_width = 28
n_visible = mnist_width * mnist_width
n_hidden_1 = 500
n_hidden_2 = 300
n_hidden_3 = 150
n_hidden_4 = 75

W_1 = tf.Variable(tf.random_uniform(shape=[n_visible, n_hidden_1]), name='W_1')
W_2 = tf.Variable(tf.random_uniform(shape=[n_hidden_1, n_hidden_2]), name='W_2')
W_3 = tf.Variable(tf.random_uniform(shape=[n_hidden_2, n_hidden_3]), name='W_3')
W_4 = tf.Variable(tf.random_uniform(shape=[n_hidden_3, n_hidden_4]), name='W_4')
b_1 = tf.Variable(tf.zeros([n_hidden_1]), name='b_1')
b_2 = tf.Variable(tf.zeros([n_hidden_2]), name='b_2')
b_3 = tf.Variable(tf.zeros([n_hidden_3]), name='b_3')
b_4 = tf.Variable(tf.zeros([n_hidden_4]), name='b_4')

W_1_prime = tf.transpose(W_1)
b_1_prime = tf.Variable(tf.zeros([n_visible]), name='b_1_prime')
W_2_prime = tf.transpose(W_2)
b_2_prime = tf.Variable(tf.zeros([n_hidden_1]), name='b_2_prime')
W_3_prime = tf.transpose(W_3)
b_3_prime = tf.Variable(tf.zeros([n_hidden_2]), name='b_3_prime')
W_4_prime = tf.transpose(W_4)
b_4_prime = tf.Variable(tf.zeros([n_hidden_3]), name='b_4_prime')

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

saver = tf.train.Saver()
saver.restore(sess,'stackedweights.sess')

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels






