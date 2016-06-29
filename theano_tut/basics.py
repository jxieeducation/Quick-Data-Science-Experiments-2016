# http://deeplearning.net/software/theano/tutorial/adding.html

import numpy as np
import theano.tensor as T
from theano import function


x = T.dscalar('x')
y = T.dscalar('y')
z = x + y

f = function([x, y], z)
f(2, 3)
z.eval({x:2, y:3})


x = T.dmatrix('x')
y = T.dmatrix('y')
z = x + y
f = function([x, y], z)
f([[1, 2], [3, 4]], [[10, 20], [30, 40]])
f(np.array([[1, 2], [3, 4]]), np.array([[10, 20], [30, 40]]))


n = 10
m = 20
X = T.arange(n * m).reshape((n, m))
u = T.arange(0, n * m, m).reshape((n, 1))
r = X - u
r.eval()


a = T.vector()
b = T.vector()
out = a ** 2 + b ** 2 + 2 * a * b
f = function([a, b], out)
n = np.array([1, 1])
f(n, n)
