import numpy as np
from math import log
import matplotlib.pyplot as plt

y_t = 0.6

x_seq = np.arange(0.05,0.95,0.01)
y_seq = []

for y_p in x_seq:
	loss = y_t * log(y_p) + (1 - y_t) * log(1 - y_p)
	y_seq += [loss]

plt.plot(x_seq, y_seq, 'ro')
plt.show()

