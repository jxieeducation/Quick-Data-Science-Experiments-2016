import numpy as np
from math import log
import matplotlib.pyplot as plt

y_t = 0.6

x_seq = np.arange(0.05,0.95,0.01)
ll_seq = []
mse_seq = []

for y_p in x_seq:
	ll_loss = y_t * log(y_p) + (1 - y_t) * log(1 - y_p)
	ll_seq += [ll_loss]
	mse_loss = -(y_t - y_p) * (y_t - y_p)
	mse_seq += [mse_loss]

plt.plot(x_seq, ll_seq, 'ro')
plt.plot(x_seq, mse_seq, 'b')
plt.show()

