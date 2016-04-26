import input_data
from keras.utils import np_utils

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L


class MNISTNet(Chain):
	def __init__(self):
		super(MNISTNet, self).__init__(
			l1 = L.Linear(28 * 28, 100),
			l2 = L.Linear(100, 10)
		)

	def __call__(self, x):
		h = F.relu(self.l1(x))
		return self.l2(h)

class Classifier(Chain):
	def __init__(self, predictor):
		super(Classifier, self).__init__(predictor=predictor)

	def __call__(self, x, t):
		y = self.predictor(x)
		self.loss = F.softmax_cross_entropy(y, t)
		self.accuracy = F.accuracy(y, t)
		return self.loss

trY, teY = np_utils.probas_to_classes(trY).astype(np.int32), np_utils.probas_to_classes(teY).astype(np.int32)

# net = MNISTNet()
# x = chainer.Variable(trX)
# print net(x)

model = L.Classifier(MNISTNet())
optimizer = optimizers.SGD()
optimizer.setup(model)

# x = chainer.Variable(trX)
# t = chainer.Variable(trY)
# print model(x, t)

teX = chainer.Variable(teX)
teY = chainer.Variable(teY)

totalSize = trX.shape[0]
batchSize = 100

for epoch in range(20):
	print('epoch: %d, loss: %f, accuracy: %f' % (epoch, model(teX, teY).data, model.accuracy.data))
	indexes = np.random.permutation(totalSize)
	for i in range(0, totalSize, batchSize):
		x = Variable(trX[indexes[i : i + batchSize]])
		t = Variable(trY[indexes[i : i + batchSize]])
		model.zerograds()
		loss = model(x, t)
		loss.backward()
		optimizer.update()

