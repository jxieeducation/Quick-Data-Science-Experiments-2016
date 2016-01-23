from gensim.models import Doc2Vec
from sklearn.manifold import TSNE
import random
import numpy as np

model = Doc2Vec.load('./imdb.d2v')

tsne = TSNE(n_components=2, perplexity=5, random_state=1337, verbose=3,  n_iter_without_progress=100)

# there are 12000 train_pos
samples = random.sample(range(0, 12000), 200)
h_vectors = []
for sample in samples:
	h_vectors += [model.docvecs[sample]]
# 25000 offset to negative
for sample in samples:
	h_vectors += [model.docvecs[sample + 25000]]
h_vectors = np.array(h_vectors)
l_vectors = tsne.fit_transform(h_vectors)

import matplotlib.pyplot as plt
from matplotlib import offsetbox

colors = np.array([0.1 for i in range(200)] + [0.7 for i in range(200)])

plt.scatter(l_vectors[:,0], l_vectors[:,1], c=colors)
plt.show()
