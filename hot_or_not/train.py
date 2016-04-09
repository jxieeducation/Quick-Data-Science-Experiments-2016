from __future__ import print_function
import numpy as np
np.random.seed(1337)

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from scipy.io import loadmat
from utils import *
from sklearn.cross_validation import train_test_split

batch_size = 16
nb_epoch = 10

img_rows, img_cols = 32, 32
nb_filters = 48
nb_pool = 2
nb_conv = 3

X, y = loadDataset()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1337)


model = Sequential()

model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                        border_mode='valid',
                        input_shape=(3, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters, nb_conv, nb_conv))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(32))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adadelta')
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          show_accuracy=False, verbose=1, validation_data=(X_test, y_test))

score = model.evaluate(X_test, y_test, show_accuracy=True, verbose=0)
print('Test score:', score[0])
