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
def index2Char(index):
	return vocab[index]
def sentence2listIndex(sentence):
	return [char2Index(char) for char in sentence]
def listIndex2Sentence(list_of_index):
	return ''.join([index2Char(index) for index in list_of_index])

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

	def predict_char(self, cur_char):
		predicted = self.__call__(cur_char)
		softmax_predicted = F.softmax(predicted)
		selected = np.argmax(softmax_predicted.data)
		selected_var = Variable(np.array([selected], dtype=np.int32))
		return selected_var

	# come --> [14, 26, 24, 16]
	def generateSentence(self, start_indexes=[14, 26, 24, 16], length=20):
		self.reset_state()
		my_list_var = []
		curr = None
		for index in start_indexes:
			curr = Variable(np.array([index], dtype=np.int32))
			my_list_var += [curr]
			self.predict_char(curr)
		for i in range(length):
			predicted = self.predict_char(curr)
			my_list_var += [predicted]
			curr = predicted
		my_list = [elem.data.item() for elem in my_list_var]
		return my_list


class Classifier(Chain):
	def __init__(self, predictor):
		super(Classifier, self).__init__(predictor=predictor)

	def __call__(self, x, t):
		y = self.predictor(x)
		self.loss = F.softmax_cross_entropy(y, t)
		self.accuracy = F.accuracy(y, t)
		return self.loss

def sentenceToChainerFormat(sentence):
	# print sentence[:len(sentence)-1], sentence[1:]
	x = sentence2listIndex(sentence[:len(sentence)-1])
	y = sentence2listIndex(sentence[1:])
	x_var_list = []
	y_var_list = []
	for index in x:
		x_var_list += [Variable(np.array([index], dtype=np.int32))]
	for index in y:
		y_var_list += [Variable(np.array([index], dtype=np.int32))]
	return x_var_list, y_var_list

rnn = RNN()
model = Classifier(rnn)
optimizer = optimizers.SGD()
optimizer.setup(model)

totalSize = len(lines) / 100 * 100
batchSize = 10

# print listIndex2Sentence(rnn.generateSentence())
for epoch in range(20):
	print('\n\nepoch: %d' % (epoch))
	indexes = np.random.permutation(totalSize)
	for i in range(0, totalSize, batchSize):
		if i % 1000 == 0:
			print 'at %d, generated: %s' % (i, listIndex2Sentence(rnn.generateSentence(start_indexes=sentence2listIndex("you are"))))
		for sentenceId in range(i, i + batchSize):
			rnn.reset_state()
			model.zerograds()
			x_list, y_list = sentenceToChainerFormat(lines[sentenceId])
			loss = 0
			for i in range(len(x_list)):
				loss += model(x_list[i], y_list[i])
			loss.backward()
			optimizer.update()
