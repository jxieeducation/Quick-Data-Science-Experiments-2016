import numpy as np 
from minisom import MiniSom

data = np.genfromtxt('isolet1+2+3+4.data', delimiter=',')

label = data[:,617]
data = data[:,0:617]
data = np.apply_along_axis(lambda x: x/np.linalg.norm(x),1,data)

som = MiniSom(10,10,617,sigma=1.0, learning_rate=0.5)

som.random_weights_init(data)
original_error = som.quantization_error(data)
print original_error

som.train_random(data, 5000)
print som.quantization_error(data)

### graphing
from pylab import plot,axis,show,pcolor,colorbar,bone
import random 

indexes = random.sample(range(0, len(label)), 500)
graph_target = label[indexes]
graph_data = data[indexes,]

t = np.zeros(len(graph_target),dtype=int)
# everything starts as 0
t[graph_target == 12] = 1
t[graph_target == 2] = 2

markers = ['o','s','D']
colors = ['r','g','b']
for cnt,xx in enumerate(graph_data):
	w = som.winner(xx)
	plot(w[0]+.5,w[1]+.5,markers[t[cnt]],markerfacecolor='None', markeredgecolor=colors[t[cnt]],markersize=12,markeredgewidth=2)

axis([0,som.weights.shape[0],0,som.weights.shape[1]])
show()


