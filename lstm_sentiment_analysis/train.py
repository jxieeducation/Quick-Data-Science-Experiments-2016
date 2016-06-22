import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils, FunctionSet
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer import serializers
import sys
import numpy as np
from net import LSTMNet

net = LSTMNet(15440)
optimizer = optimizers.SGD(lr=0.05)
optimizer.setup(net)

count = 0
for epoch in range(5):
	f = open('data/train.txt', 'r')
	for line in f:
		if line == "":
			continue
		count += 1
		label, paragraph = line.split('\t')
		label = int(label)
		paragraph = [int(word) for word in paragraph.split(',')]
		net.zerograds()
		loss = net.train(paragraph, label)
		print "count: %d, error: %f" % (count, loss.data)
		loss.backward()
		optimizer.update()
	serializers.save_npz("data/models/%d.model" % epoch, net)
	serializers.save_npz("data/models/%d.optimizer" % epoch, optimizer)
