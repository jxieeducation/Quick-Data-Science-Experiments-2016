import numpy as np
import input_data
from sklearn import decomposition
from sklearn.neighbors import KNeighborsClassifier

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels

print "starting PCA"

pca = decomposition.PCA(n_components=36)
pca.fit(trX)
trX = pca.transform(trX)
teX = pca.transform(teX)

print "starting KNN"

neigh = KNeighborsClassifier(n_neighbors=10)
neigh.fit(trX, trY)
# teY_prediction = neigh.kneighbors(teX)
# this part needs to be finished...
