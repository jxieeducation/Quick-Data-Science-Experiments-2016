import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L

l = L.LSTM(100, 50)

l.reset_state()
x = Variable(np.random.randn(10, 100).astype(np.float32))
y = l(x)
x2 = Variable(np.random.randn(10, 100).astype(np.float32))
y2 = l(x2)

class RNN(Chain):
    def __init__(self):
        super(RNN, self).__init__(
            embed=L.EmbedID(1000, 100),
            mid=L.LSTM(100, 50),
            out=L.Linear(50, 1000), 
        )

    def reset_state(self):
        self.mid.reset_state()

    def __call__(self, cur_word):
        # Given the current word ID, predict the next word.
        x = self.embed(cur_word)
        h = self.mid(x)
        y = self.out(h)
        return y

rnn = RNN()
model = L.Classifier(rnn)
optimizer = optimizers.SGD()
optimizer.setup(model)

def compute_loss(x_list):
    loss = 0
    for cur_word, next_word in zip(x_list, x_list[1:]):
        loss += model(cur_word, next_word)
    return loss

rnn.reset_state()
model.zerograds()
loss = compute_loss(x_list) # we dont have x_list Q.Q
loss.backward()
optimizer.update()

# OR
# rnn.reset_state()
# optimizer.update(compute_loss, x_list)




