from kvxopt import matrix, solvers
from kvxopt.solvers import SolverResult

c = matrix([-4.0, -5.0])
G = matrix([[2.0, 1.0, -1.0, 0.0], [1.0, 2.0, 0.0, -1.0]])
h = matrix([3.0, 3.0, 0.0, 0.0])

lp_res = solvers.lp(c, G, h)
qp_res = solvers.qp(matrix([[2.0, 0.0], [0.0, 2.0]]), c)

_: SolverResult = lp_res
_: SolverResult = qp_res
