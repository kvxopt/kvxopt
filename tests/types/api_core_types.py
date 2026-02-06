from kvxopt import div, matrix, max, min, mul, normal, sparse, spdiag, spmatrix, uniform

v = matrix([1.0, 2.0, 3.0])
w = matrix([3.0, -2.0, -1.0])

m1 = normal(4, 2)
m2 = uniform(4, 2)

mx = max(v, w)
mn = min(v, w)
prd = mul(v, w)
q = div(v, 2.0)

s = spmatrix([1.0, 2.0], [0, 1], [0, 1], (2, 2))
sm = sparse([s, s])
sd = spdiag(v)

_m1: matrix = m1
_m2: matrix = m2
_mx: matrix = mx
_mn: matrix = mn
_prd: matrix = prd
_q: matrix = q
_sm: spmatrix = sm
_sd: spmatrix = sd
