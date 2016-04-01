import numpy as np
import input_data
import tensorflow as tf
import math
import matplotlib
import matplotlib.pyplot as plt

''' Getting the data '''
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

noisy_input = trX + .2 * np.random.random_sample((trX.shape)) - .1
# this adds noise, becomes denoising autoencoder
output = trX
scaled_input = np.divide((noisy_input-noisy_input.min()), (noisy_input.max()-noisy_input.min()))
scaled_output = np.divide((output-output.min()), (output.max()-output.min()))
input_data = scaled_input * 2 - 1
output_data = scaled_output * 2 - 1


''' Defining TensorFlow '''
n_samp, n_input = input_data.shape 
n_hidden = 8

x = tf.placeholder("float", [None, n_input])
# Weights and biases to hidden layer
Wh = tf.Variable(tf.random_uniform((n_input, n_hidden), -1.0 / math.sqrt(n_input), 1.0 / math.sqrt(n_input)))
bh = tf.Variable(tf.zeros([n_hidden]))
h = tf.nn.tanh(tf.matmul(x,Wh) + bh)
# Weights and biases to hidden layer
Wo = tf.transpose(Wh) # tied weights
bo = tf.Variable(tf.zeros([n_input]))
y = tf.nn.tanh(tf.matmul(h,Wo) + bo)
# Objective functions
y_ = tf.placeholder("float", [None,n_input])
cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)))
# meansq = tf.reduce_mean(tf.square(y_-y))
# train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)
train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)

''' Executing TensorFlow '''
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

n_rounds = 4000
batch_size = min(512, n_samp)

for i in range(n_rounds):
    sample = np.random.randint(n_samp, size=batch_size)
    batch_xs = input_data[sample][:]
    batch_ys = output_data[sample][:]
    sess.run(train_step, feed_dict={x: batch_xs, y_:batch_ys})
    if i % 100 == 0:
        print i, sess.run(cross_entropy, feed_dict={x: batch_xs, y_:batch_ys})
        print sess.run(y, feed_dict={x: np.array([batch_xs[0]])})[0][:10]

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
