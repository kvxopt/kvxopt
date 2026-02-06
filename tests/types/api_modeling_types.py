from kvxopt import matrix
from kvxopt.modeling import dot, max, op, sum, variable

x = variable(2, "x")
y = variable(2, "y")
A = matrix([[1.0, 0.0], [0.0, 1.0]])
b = matrix([1.0, 1.0])

c1 = A * x <= b
c2 = x >= 0
problem = op(dot(matrix([1.0, 1.0]), x), [c1, c2])
expr = sum(max(0, x - y))
