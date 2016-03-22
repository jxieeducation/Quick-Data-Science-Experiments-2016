import numpy as np 

"""
outer(u, v) = u * v.T
u -> (5, 1)
v -> (3, 1)
u * v.T -> (5, 1) * (1, 3) -> (5, 3)
"""

a = np.array([1, 1, 1, 1, 1])
b = np.array([1, 1, 1])
print a, b
print np.outer(a, b)

a = np.array([1, 0, 0, 0, 0])
b = np.array([1, 1, 1])
print a, b
print np.outer(a, b)

