import numpy as np
from chainer import Variable, FunctionSet
import chainer.functions as F
import chainer.links as L

class CharRNN(FunctionSet):

    def __init__(self, n_vocab, n_units):
        super(CharRNN, self).__init__(
            embed = F.EmbedID(n_vocab, n_units),
            # l1_x = L.Linear(n_units, 4*n_units),
            # l1_h = L.Linear(n_units, 4*n_units),
            # l2_h = L.Linear(n_units, 4*n_units),
            # l2_x = L.Linear(n_units, 4*n_units),
            l1   = L.LSTM(n_units, n_units),
            l2   = L.LSTM(n_units, n_units),
            l3   = L.Linear(n_units, n_vocab),
        )
        for param in self.parameters:
            param[:] = np.random.uniform(-0.08, 0.08, param.shape)

    def forward_one_step(self, x_data, y_data, train=True, dropout_ratio=0.5):
        x = Variable(x_data, volatile=not train)
        t = Variable(y_data, volatile=not train)

        h0      = self.embed(x)
        h1      = self.l1(F.dropout(h0, ratio=dropout_ratio, train=train))
        h2      = self.l2(F.dropout(h1, ratio=dropout_ratio, train=train))
        y       = self.l3(F.dropout(h2, ratio=dropout_ratio, train=train))

        if train:
            return F.softmax_cross_entropy(y, t)
        else:
            return F.softmax(y)
