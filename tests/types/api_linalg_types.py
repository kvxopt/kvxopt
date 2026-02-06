from kvxopt import blas, lapack, matrix

A = matrix([[2.0, 1.0], [1.0, 2.0]])
B = matrix([[1.0, 0.0], [0.0, 1.0]])
x = matrix([1.0, 2.0])
y = matrix([0.0, 0.0])

_ = blas.dot(x, x)
blas.axpy(x, y)
blas.gemm(A, B)
_ = lapack.potrf(A)
_ = lapack.posv(A, x)
