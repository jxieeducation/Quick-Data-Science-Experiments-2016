import numpy as np
import theano
import theano.tensor as T
import lasagne
import gzip
import pickle

np.random.seed(42)

train, val, test = pickle.load(gzip.open('mnist.pkl.gz'))

X_train, y_train = train
X_val, y_val = val

print X_train.shape

def batch_gen(X, y, N):
    while True:
        idx = np.random.choice(len(y), N)
        yield X[idx].astype('float32'), y[idx].astype('int32')

l_in1 = lasagne.layers.InputLayer((None, 784))
l_out1 = lasagne.layers.DropoutLayer(l_in1, p=0.35)
l_out2 = lasagne.layers.DenseLayer(l_out1, num_units=100, nonlinearity=lasagne.nonlinearities.sigmoid)
l_out = lasagne.layers.DenseLayer(l_out1, num_units=10, nonlinearity=lasagne.nonlinearities.softmax)

X_sym = T.matrix()
y_sym = T.ivector()

output = lasagne.layers.get_output(l_out, X_sym)
pred = output.argmax(-1)

loss = T.mean(lasagne.objectives.categorical_crossentropy(output, y_sym))
acc = T.mean(T.eq(pred, y_sym))

params = lasagne.layers.get_all_params(l_out)
# print params

grad = T.grad(loss, params)
# updates = lasagne.updates.sgd(grad, params, learning_rate=0.05)
updates = lasagne.updates.rmsprop(grad, params, learning_rate=0.01)
print(updates)

f_train = theano.function([X_sym, y_sym], [loss, acc], updates=updates)
f_val = theano.function([X_sym, y_sym], [loss, acc])
f_predict = theano.function([X_sym], pred)



BATCH_SIZE = 64
N_BATCHES = len(X_train) // BATCH_SIZE
N_VAL_BATCHES = len(X_val) // BATCH_SIZE
train_batches = batch_gen(X_train, y_train, BATCH_SIZE)
val_batches = batch_gen(X_val, y_val, BATCH_SIZE)

for epoch in range(100):
    train_loss = 0
    train_acc = 0
    for _ in range(N_BATCHES):
        X, y = next(train_batches)
        loss, acc = f_train(X, y)
        train_loss += loss
        train_acc += acc
    train_loss /= N_BATCHES
    train_acc /= N_BATCHES

    val_loss = 0
    val_acc = 0
    for _ in range(N_VAL_BATCHES):
        X, y = next(val_batches)
        loss, acc = f_val(X, y)
        val_loss += loss
        val_acc += acc
    val_loss /= N_VAL_BATCHES
    val_acc /= N_VAL_BATCHES
    
    print('Epoch {}, Train (val) loss {:.03f} ({:.03f}) ratio {:.03f}'.format(
            epoch, train_loss, val_loss, val_loss/train_loss))
    print('Train (val) accuracy {:.03f} ({:.03f})'.format(train_acc, val_acc))


weights = l_out.W.get_value()
print(weights.shape)

