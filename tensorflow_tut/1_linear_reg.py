import tensorflow as tf
import numpy as np

trX = np.linspace(-1, 1, 1000)
trY = 2 * trX + np.random.randn(*trX.shape) * 0.33

X = tf.placeholder("float")
Y = tf.placeholder("float")

def model(X, w):
    return tf.mul(X, w)

w = tf.Variable(0.0, name="weights")
y_model = tf.mul(X, w)

cost = (tf.pow(Y-y_model, 2))

train_op = tf.train.GradientDescentOptimizer(0.0001).minimize(cost)

sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

for i in range(100):
    print str(i) + ": " + str(sess.run(w))
    for (x, y) in zip(trX, trY): 
        sess.run(train_op, feed_dict={X: x, Y: y})

