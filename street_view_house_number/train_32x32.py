from __future__ import print_function
import numpy as np
np.random.seed(1337)

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from scipy.io import loadmat
from util import *
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

batch_size = 128
nb_classes = 10
nb_epoch = 2

img_rows, img_cols = 32, 32
nb_filters = 32
nb_pool = 2
nb_conv = 3

matfile = loadmat('train_32x32.mat')
X = matfile['X'].astype('float32')
X = reshapeToKeras2D(X)
X /= 255
y = matfile['y']
y = y - 1 # y originally is 1-10, convert to 0-9 first
y = np_utils.to_categorical(y, nb_classes)
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
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adadelta')
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          show_accuracy=True, verbose=1, validation_data=(X_test, y_test))

score = model.evaluate(X_test, y_test, show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

y_hat = model.predict(X_test)
y_test_classes = np_utils.categorical_probas_to_classes(y_test)
y_hat_classes = np_utils.categorical_probas_to_classes(y_hat)
print confusion_matrix(y_test_classes, y_hat_classes)

#showWrongOnes(X_test, y_test_classes, y_hat_classes, 0, 5)

