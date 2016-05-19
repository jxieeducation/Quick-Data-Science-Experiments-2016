import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils, FunctionSet
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.functions.loss.vae import gaussian_kl_divergence
import six

class SimpleNet():
    def __init__(self):
        n_in = 1
        n_hidden_1 = 5
        n_hidden_2 = 5
        self.model = FunctionSet(
        	en1=L.Linear(n_in, n_hidden_1),
        	en2_mu=L.Linear(n_hidden_1, n_hidden_2),
        	en2_var=L.Linear(n_hidden_1, n_hidden_2),
        	de1=L.Linear(n_hidden_2, n_hidden_1),
        	de2=L.Linear(n_hidden_1, n_in)
        )
        self.optimizer = optimizers.Adam()
        self.optimizer.setup(self.model.collect_parameters())

    def encode(self, x_var):
    	h1 = F.tanh(self.model.en1(x_var))
    	mu = self.model.en2_mu(h1)
    	var = self.model.en2_var(h1)
    	return mu, var

    def decode(self, z, sigmoid=True):
    	h1 = F.tanh(self.model.de1(z))
    	h2 = self.model.de2(h1)
    	if sigmoid:
    		return F.sigmoid(h2)
    	return h2

    def cost(self, x_var, C=1.0, k=1):
    	mu, ln_var = self.encode(x_var)
    	batchsize = len(mu.data)
    	rec_loss = 0
    	for l in six.moves.range(k):
    		z = F.gaussian(mu, ln_var)
    		rec_loss += F.bernoulli_nll(x_var, self.decode(z, sigmoid=False)) \
    		/ (k * batchsize)
    	self.rec_loss = rec_loss
    	self.loss = self.rec_loss + C * gaussian_kl_divergence(mu, ln_var) / batchsize
    	return self.loss

net = SimpleNet()

def generateTrainingData():
	mu = [0, 10, 20]
	sigma = [0.3, 1.1, 2.2]

	gen_batch = 5000
	myTrainingData = None
	for index in range(len(mu)):
		s = np.random.normal(mu[index], sigma[index], gen_batch).reshape(gen_batch, 1)
		if myTrainingData is None:
			myTrainingData = s
		else:
			myTrainingData = np.concatenate([myTrainingData, s])
	myTrainingData = np.random.permutation(myTrainingData)
	myTrainingData = myTrainingData.astype(np.float32)
	return myTrainingData

for i in range(20):
	net.model.zerograds()
	train = generateTrainingData()
	loss = net.cost(chainer.Variable(train))
	print "epoch: %s, loss: %s" % (i, loss.data)
	loss.backward()
	net.optimizer.update()

def tryEncode(net, mu=10, sigma=1.1):
	data = np.random.normal(mu, sigma, 1).reshape(1, 1).astype(np.float32)
	print "\n\n\n"
	print data
	en_mu, en_var = net.encode(chainer.Variable(data))
	print en_mu.data
	print en_var.data

def tryEncode2(net, number=10):
	data = np.array([number]).reshape(1, 1).astype(np.float32)
	print "\n\n\n"
	print data
	en_mu, en_var = net.encode(chainer.Variable(data))
	print en_mu.data
	print en_var.data

tryEncode2(net, 0)
tryEncode2(net, 0.3)
tryEncode2(net, 10)
tryEncode2(net, 11)
tryEncode2(net, 20)
tryEncode2(net, 21)
