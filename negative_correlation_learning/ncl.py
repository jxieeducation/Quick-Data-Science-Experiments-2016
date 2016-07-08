import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils, FunctionSet
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

class NN(chainer.Chain):
	def __init__(self):
		super(NN, self).__init__(
			l1=L.Linear(784, 10)
		)

	def __call__(self, X):
		return F.softmax(self.l1(X))

class NCL(chainer.Chain):
	def __init__(self, ensemble_size=3, lambda_val=0.5):
		self.ensemble_size = ensemble_size
		self.ensemble = [NN() for i in range(ensemble_size)]
		self.lambda_val = lambda_val

	def __call__(self, X):
		res = None
		predictions = []
		for nn in self.ensemble:
			pred = nn(X)
			predictions += [pred]
			if res is not None:
				res += pred
			else: 
				res = pred
		res = res / self.ensemble_size
		return res, predictions

	def train(self, X, y):
		ensemble_pred,_ = self.__call__(X)
		losses = []
		for nn in self.ensemble:
			pred = nn(X)
			loss = F.mean_squared_error(pred, y) 
			p = - F.sum((pred - ensemble_pred) * (pred - ensemble_pred))
			# print p.data
			loss += self.lambda_val * p
			losses += [loss]
		return losses

import input_data
mnist = input_data.read_data_sets("/tmp/MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
trY, teY = trY.astype(np.float32), teY.astype(np.float32)


ncl = NCL(ensemble_size=4, lambda_val=0.01)
my_optimizers = []
for nn in ncl.ensemble:
	optimizer = optimizers.Adam()
	optimizer.setup(nn)
	my_optimizers += [optimizer]

totalSize = trX.shape[0]
batchSize = 128
for epoch in range(5):
    print('epoch: %d,' % (epoch))
    _, predictions = ncl(chainer.Variable(trX[[1]]))
    print np.correlate(predictions[0].data[0], predictions[1].data[0])
    indexes = np.random.permutation(totalSize)
    for i in range(0, totalSize, batchSize):
        X = chainer.Variable(trX[indexes[i : i + batchSize]])
        y = chainer.Variable(trY[indexes[i : i + batchSize]])
        for i in range(len(ncl.ensemble)):
            my_optimizers[i].zero_grads()
        losses = ncl.train(X, y)
        for i in range(len(losses)):
            optimizer = my_optimizers[i]
            loss = losses[i]
            loss.backward()
            optimizer.update()
            # print loss.data


