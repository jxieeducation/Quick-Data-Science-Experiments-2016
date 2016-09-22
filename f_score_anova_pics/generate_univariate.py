from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1, 1)

mean, var, skew, kurt = norm.stats(moments='mvsk')

x = np.linspace(0, 10, 100)

# 2, 0.1; 8, 0.1
# 4.8, 0.1; 5.2, 0.1
# 4.8, 1; 5.2, 1
# 2, 1; 8, 1

ax.plot(x, norm.pdf(x, 2, 1), 'r-', lw=5, alpha=0.6, label='norm pdf')
ax.plot(x, norm.pdf(x, 8, 1), 'b-', lw=5, alpha=0.6, label='norm pdf')

plt.show()
fig.savefig('graph.png')
