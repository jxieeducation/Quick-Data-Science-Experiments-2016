''' 
Adding image augmentation this time
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
from keras.preprocessing.image import ImageDataGenerator


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
    num_examples = 5000
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
    X = np.array(X)
    # X /= 255.0 # no...
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
    datagen.fit(X_train)
    model.fit_generator(datagen.flow(X_train, y_train,
                        batch_size=32),
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=20,
                        validation_data=(X_test, y_test))

    out = model.predict(X_test)
    print accuracy_score(np.argmax(y_test, axis=1), np.argmax(out, axis=1))
