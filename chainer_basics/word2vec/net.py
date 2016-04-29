import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils, FunctionSet
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

# CBOW --> given context, predict vector
'''
1. EmbedID()
2. vector for X = sum(all context embeddings)
'''
# Skip-gram --> given vector, predict context
'''
1. EmbedID()
2. loss for X = sum(all context vectors' distance from X)
'''

class MLP(Chain):
    def __init__(self):
        super(MLP, self).__init__(
            l1=L.Linear(784, 100),
            l2=L.Linear(100, 100),
            l3=L.Linear(100, 10),
        )

    def __call__(self, x):
        h1 = F.relu(self.l1(x))
        h2 = F.relu(self.l2(h1))
        y = self.l3(h2)
        return y



