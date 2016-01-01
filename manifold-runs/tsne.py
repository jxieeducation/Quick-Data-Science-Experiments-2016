from sklearn.manifold import TSNE
import numpy as np 
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from matplotlib import figure

print "original dim: " + str(iris.data.shape)
iris = load_iris()
print "transforming into 2-dim"
X_tsne = TSNE(learning_rate=100).fit_transform(iris.data)

plt.figure(figsize=(10,5))
plt.subplot(121)
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=iris.target)
plt.show()
