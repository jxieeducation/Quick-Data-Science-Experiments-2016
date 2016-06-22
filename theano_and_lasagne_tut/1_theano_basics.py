import theano
import theano.tensor as T

x = T.scalar()
y = 3*(x**2) + 1

theano.pprint(y)
theano.printing.debugprint(y)

print y.eval({x: 2})
f = theano.function([x], y)
print f(2)


X = T.vector()
X = T.matrix()
X = T.tensor3()
X = T.tensor4()

