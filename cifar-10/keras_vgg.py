''' THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 

export PYTHONPATH=$PYTHONPATH:/usr/local/lib/python2.7/dist-packages/keras
'''

import sys
sys.path.append("/usr/local/lib/python2.7/dist-packages")
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, RMSprop
from skimage.transform import resize
from skimage.io import imread
from skimage import img_as_ubyte
import numpy as np
from skdata import cifar10
from sklearn.cross_validation import train_test_split
from keras.utils import np_utils
from sklearn.metrics import accuracy_score


def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model

def pop_layer(model):
    if not model.outputs:
        raise Exception('Sequential model cannot be popped: model is empty.')

    model.layers.pop()
    if not model.layers:
        model.outputs = []
        model.inbound_nodes = []
        model.outbound_nodes = []
    else:
        model.layers[-1].outbound_nodes = []
        model.outputs = [model.layers[-1].output]
    model.built = False

if __name__ == "__main__":
    model = VGG_16('vgg16_weights.h5')
    # model.layers.pop()
    pop_layer(model)
    for l in model.layers:
        l.trainable = False
    model.add(Dense(10, activation='softmax'))
    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')

    c10 = cifar10.dataset.CIFAR10()
    c10.fetch(download_if_missing=True)
    num_examples = 16000
    labels = [m['label'] for m in c10.meta][:num_examples]
    y = c10._labels[:num_examples]
    y = np_utils.to_categorical(y, 10)
    X = [] 
    # c10._pixels.shape[0]
    for i in range(num_examples): # (60000, 32, 32, 3)
        if i % 100 == 0:
            print "loading data: %d / %d" % (i, num_examples)
        im = c10._pixels[i]
        im = resize(im, (224, 224))
        im = img_as_ubyte(im)
        im = im.astype(np.float32)
        im[:,:,0] -= 103.939 # minus the mean pixels
        im[:,:,1] -= 116.779
        im[:,:,2] -= 123.68
        im = im.transpose((2,0,1))
        X += [im]

    # X /= 255.0 # no...
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    model.fit(X_train, y_train, batch_size=32, nb_epoch=10, validation_data=(X_test, y_test), metrics=["accuracy"])

    out = model.predict(X_test)
    print accuracy_score(np.argmax(y_test, axis=1), np.argmax(out, axis=1))


