import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils, FunctionSet
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L


class MNISTNet():
    def __init__(self):
        n_in = 28 * 28
        n_hidden = 100
        self.model = FunctionSet(
            encode=F.Linear(n_in, n_hidden),
            decode=F.Linear(n_hidden, n_in)
        )
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model.collect_parameters())

    def encode(self, x_var):
        return F.sigmoid(self.model.encode(x_var))

    def decode(self, x_var):
        return F.sigmoid(self.model.decode(x_var))

    def predict(self, x_data):
        x = Variable(x_data)
        p = self.encode(x)
        return p.data

    def cost(self, x_data, dropout=True):
        x = Variable(x_data)
        t = Variable(x_data)
        if dropout:
            x_n = F.dropout(x, ratio=0.4)
        else: 
        	x_n = x
        h = self.encode(x_n)
        y = self.decode(h)
        return F.mean_squared_error(y, t)

    def train(self, x_data):
        self.optimizer.zero_grads()
        loss = self.cost(x_data)
        loss.backward()
        self.optimizer.update()
        return float(loss.data)

net = MNISTNet()

totalSize = trX.shape[0]
batchSize = 100
for epoch in range(20):
    print('epoch: %d, loss: %f' % (epoch, net.cost(teX, dropout=False).data))
    indexes = np.random.permutation(totalSize)
    for i in range(0, totalSize, batchSize):
    	net.train(trX[indexes[i : i + batchSize]])
