import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils, FunctionSet
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

class LSTMNet(Chain):
	def __init__(self, dim):
		super(LSTMNet, self).__init__(
			embed=L.EmbedID(dim, 64),
			l1=L.LSTM(64, 64),
			l2=L.LSTM(64, 64),
			full1=L.Linear(64, 16),
			full2=L.Linear(16, 2)
		)
		for param in self.params():
			param = param.data
			param[:] = np.random.uniform(-0.08, 0.08, param.shape)

	def reset_state(self):
		self.l1.reset_state()
		self.l2.reset_state()

	def embed_word(self, x):
		embed_vec = self.embed(x)
		h1 = self.l1(embed_vec)
		h2 = self.l2(h1)
		return h2

	def forward(self, paragragh_int_list):
		for word in paragragh_int_list:
			if word == 15439: # im not sure about what this means
				continue
			x_var = chainer.Variable(np.array([word], dtype=np.int32))
			out = self.embed_word(x_var)

		o1 = self.full1(out)
		o2 = self.full2(o1)
		return o2

	def train(self, paragragh_int_list, label):
		self.reset_state()
		y_pred = self.forward(paragragh_int_list)
		loss = F.softmax_cross_entropy(y_pred, chainer.Variable(np.array([label], dtype=np.int32)))
		return loss


# test = [1, 2, 3, 4, 5]
# net = LSTMNet(10)
# print net.forward(test).data
# print net.train(test, 1).data
