from scipy.io import loadmat
from utils import *
from sklearn.cross_validation import train_test_split
import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils, FunctionSet
import chainer.functions as F
import chainer.links as L

class Alex(chainer.Chain):
    def __init__(self):
        super(Alex, self).__init__(
            conv1=L.Convolution2D(3,  32, 4, stride=4),
            conv2=L.Convolution2D(32, 16,  3, pad=2),
            conv3=L.Convolution2D(16, 16,  2, pad=1),
            fc4=L.Linear(256, 32),
            fc5=L.Linear(32, 1)
        )
    def __call__(self, x, train=True):
        h = F.max_pooling_2d(F.relu(
            F.local_response_normalization(self.conv1(x))), 3, stride=2)
        h = F.max_pooling_2d(F.relu(
            F.local_response_normalization(self.conv2(h))), 3, stride=2)
        h = F.relu(self.conv3(h))	
        h = F.dropout(F.relu(self.fc4(h)), train=train)
        h = self.fc5(h)
        return h

    def train(self, x, t, train=True):
    	h = self.__call__(x, train=train)
        loss = chainer.functions.mean_squared_error(h, t)
        return loss

print "Loading dataset"
X, y = loadDataset()
X, y = X.astype(np.float32), y.astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1337)
y_train.shape, y_test.shape = (y_train.shape[0], 1), (y_test.shape[0], 1)

alex = Alex()
optimizer = optimizers.RMSprop(lr=0.001, alpha=0.9)
# optimizer = optimizers.RMSprop()
optimizer.setup(alex)
for i in range(150):
	optimizer.zero_grads()
	alex.zerograds()
	loss = alex.train(chainer.Variable(X_train), chainer.Variable(y_train))
	eval_loss = F.mean_squared_error(alex(chainer.Variable(X_test)), chainer.Variable(y_test))
	print "epoch: %d, eval, train loss: %f, eval loss: %f" % (i, loss.data, eval_loss.data)
	loss.backward()
	optimizer.clip_grads(2.0)
	optimizer.update()
