'''
http://docs.chainer.org/en/stable/tutorial/recurrentnet.html
https://github.com/pfnet/chainer/blob/master/examples/ptb/net.py
https://github.com/karpathy/char-rnn
https://github.com/yusuketomoto/chainer-char-rnn/blob/master/train.py
http://karpathy.github.io/2015/05/21/rnn-effectiveness/
'''

###### preprocessing data ######
f = open('shakespeare.txt')
raw = f.read().lower()
vocab = list(set(list(raw)))
content = raw.split('\n')
lines = []
for line in content:
	if line != '' and len(line.split()) > 4:
		lines += [line]

def char2Index(char):
	return vocab.index(char)
def index2Index(index):
	return vocab[index]

import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils, FunctionSet
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

class RNN(Chain):
	def __init__(self):
		super(RNN, self).__init__(
			embed=L.EmbedID(len(vocab), 100),
			mid=L.LSTM(100, 50),
			out=L.Linear(50, len(vocab)),
		)

	def reset_state(self):
		self.mid.reset_state()

	def __call__(self, cur_char):
		x = self.embed(cur_char)
		h = self.mid(x)
		y = self.out(h)
		return y

	def predict(self, cur_char):
		return cur_char # replace with softmax

class Classifier(Chain):
	def __init__(self, predictor):
		super(Classifier, self).__init__(predictor=predictor)

	def __call__(self, x, t):
		y = self.predictor(x)
		self.loss = F.softmax_cross_entropy(y, t)
		self.accuracy = F.accuracy(y, t)
		return self.loss

rnn = RNN()
model = L.Classifier(rnn)
optimizer = optimizers.SGD()
optimizer.setup(model)

totalSize = len(lines)
batchSize = 100 # 100 sentences

def compute_loss(x_list):
	loss = 0
	for cur_word, next_word in zip(x_list, x_list[1:]):
		loss += model(cur_word, next_word)
	return loss

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







