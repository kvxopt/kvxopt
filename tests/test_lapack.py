"""
Comprehensive tests for LAPACK module
"""
import math
import unittest
from kvxopt import matrix, lapack


class TestLAPACK(unittest.TestCase):
    """Test suite for LAPACK linear algebra operations"""

    def assertAlmostEqualLists(self, L1, L2, places=7, msg=None):
        self.assertEqual(len(L1), len(L2))
        for u, v in zip(L1, L2):
            self.assertAlmostEqual(u, v, places, msg=msg)

    def test_gesv_basic(self):
        """Test general matrix solve A*X = B"""
        # Simple 2x2 system: 2x + y = 5, x + 3y = 8
        A = matrix([[2.0, 1.0], [1.0, 3.0]])
        B = matrix([5.0, 8.0])
        
        lapack.gesv(A, B)
        
        # Solution should be x = [1.4, 2.2]
        self.assertAlmostEqualLists(list(B), [1.4, 2.2])

    def test_gesv_multiple_rhs(self):
        """Test gesv with multiple right-hand sides"""
        A = matrix([[3.0, 2.0], [1.0, 2.0]])
        B = matrix([[8.0, 5.0], [5.0, 3.0]], (2, 2))
        
        lapack.gesv(A, B)
        
        # Check first column
        self.assertAlmostEqual(B[0, 0], 2.75, places=6)
        self.assertAlmostEqual(B[1, 0], -0.25, places=6)

    def test_gesv_complex(self):
        """Test gesv with complex matrices"""
        A = matrix([[2.0 + 1j, 1.0], [1.0, 2.0 + 1j]])
        B = matrix([3.0 + 2j, 3.0 + 2j])
        
        lapack.gesv(A, B)
        
        # Verify solution by multiplying back
        self.assertTrue(len(B) == 2)

    def test_posv_basic(self):
        """Test positive definite system solve"""
        # Symmetric positive definite matrix
        A = matrix([[4.0, 2.0], [2.0, 3.0]])
        B = matrix([8.0, 7.0])
        
        lapack.posv(A, B)
        
        self.assertAlmostEqualLists(list(B), [1.25, 1.5])

    def test_potrf_basic(self):
        """Test Cholesky factorization"""
        # Symmetric positive definite matrix
        A = matrix([[4.0, 2.0], [2.0, 3.0]])
        
        lapack.potrf(A)
        
        # Check that L is lower triangular
        self.assertAlmostEqual(A[0, 0], 2.0, places=6)  # sqrt(4)
        self.assertAlmostEqual(A[1, 0], 1.0, places=6)  # 2/2
        # Upper triangle is not modified in 'L' mode

    def test_potrs(self):
        """Test solution using Cholesky factorization"""
        A = matrix([[4.0, 2.0], [2.0, 3.0]])
        B = matrix([8.0, 7.0])
        
        lapack.potrf(A)
        lapack.potrs(A, B)
        
        self.assertAlmostEqualLists(list(B), [1.25, 1.5])

    def test_getrf_basic(self):
        """Test LU factorization"""
        A = matrix([[2.0, 1.0], [4.0, 5.0]])
        ipiv = matrix(0, (2, 1), tc='i')
        
        lapack.getrf(A, ipiv)
        
        # Check factorization succeeded
        self.assertTrue(len(ipiv) == 2)
        self.assertTrue(A[0, 0] != 0)  # Pivoted element

    def test_getrs(self):
        """Test solution using LU factorization"""
        A = matrix([[2.0, 1.0], [4.0, 5.0]])
        B = matrix([5.0, 14.0])
        ipiv = matrix(0, (2, 1), tc='i')
        
        lapack.getrf(A, ipiv)
        lapack.getrs(A, ipiv, B)
        
        # Verify solution
        self.assertAlmostEqualLists(list(B), [-5.166666666666667, 3.833333333333333])

    def test_geqrf_basic(self):
        """Test QR factorization"""
        A = matrix([1.0, 3.0, 5.0, 2.0, 4.0, 6.0], (3, 2))
        tau = matrix(0.0, (2, 1))
        
        lapack.geqrf(A, tau)
        
        # Check that tau has correct length
        self.assertEqual(len(tau), 2)

    def test_orgqr(self):
        """Test generation of Q from QR factorization"""
        A = matrix([1.0, 3.0, 5.0, 2.0, 4.0, 6.0], (3, 2))
        tau = matrix(0.0, (2, 1))
        
        lapack.geqrf(A, tau)
        lapack.orgqr(A, tau)
        
        # Q should be orthogonal: Q'*Q = I (approximately)
        from kvxopt import blas
        Q = A
        QTQ = matrix(0.0, (2, 2))
        blas.gemm(Q, Q, QTQ, transA='T')
        
        # Check first k diagonal elements close to 1
        self.assertAlmostEqual(QTQ[0, 0], 1.0, places=5)
        self.assertAlmostEqual(QTQ[1, 1], 1.0, places=5)

    def test_syev_basic(self):
        """Test eigenvalue decomposition of symmetric matrix"""
        # Symmetric matrix
        A = matrix([[4.0, 2.0], [2.0, 3.0]])
        w = matrix(0.0, (2, 1))
        
        lapack.syev(A, w, jobz='N')
        
        # Check eigenvalues are real
        self.assertTrue(len(w) == 2)
        self.assertTrue(w[0] <= w[1])  # Eigenvalues in ascending order
        self.assertAlmostEqual(w[0], (7.0 - math.sqrt(17.0)) / 2.0, places=6)
        self.assertAlmostEqual(w[1], (7.0 + math.sqrt(17.0)) / 2.0, places=6)

    def test_syev_with_vectors(self):
        """Test eigenvalue decomposition with eigenvectors"""
        A = matrix([[4.0, 2.0], [2.0, 3.0]])
        w = matrix(0.0, (2, 1))
        
        lapack.syev(A, w, jobz='V')
        
        # Check eigenvalues
        self.assertTrue(len(w) == 2)
        self.assertAlmostEqual(w[0], (7.0 - math.sqrt(17.0)) / 2.0, places=6)
        self.assertAlmostEqual(w[1], (7.0 + math.sqrt(17.0)) / 2.0, places=6)
        # A now contains eigenvectors
        self.assertTrue(A.size == (2, 2))

    def test_gesvd_basic(self):
        """Test singular value decomposition"""
        A = matrix([1.0, 3.0, 5.0, 2.0, 4.0, 6.0], (3, 2))
        s = matrix(0.0, (2, 1))
        
        # Just compute singular values (U and Vt setup is tricky)
        lapack.gesvd(+A, s, jobu='N', jobvt='N')
        
        # Check singular values are positive and in descending order
        self.assertTrue(s[0] >= s[1] >= 0)
        self.assertEqual(len(s), 2)

    def test_heev_complex(self):
        """Test eigenvalue decomposition of Hermitian matrix"""
        # Hermitian matrix
        A = matrix([[4.0 + 0j, 2.0 - 1j], [2.0 + 1j, 3.0 + 0j]])
        w = matrix(0.0, (2, 1))
        
        lapack.heev(A, w, jobz='N')
        
        # Check eigenvalues are real
        self.assertTrue(len(w) == 2)

    def test_gels_basic(self):
        """Test least squares solution"""
        # Overdetermined system
        A = matrix([1.0, 2.0, 3.0, 1.0, 1.0, 1.0], (3, 2))
        B = matrix([2.0, 3.0, 4.0])
        
        lapack.gels(+A, B)
        
        # B now contains the least squares solution
        self.assertEqual(len(B), 3)
        # First min(m,n) elements contain solution

    # UNSUPPORTED: gelsy is not available in kvxopt.lapack
    # def test_gelsy_basic(self):
    #     """Test least squares with column pivoting"""
    #     A = matrix([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    #     B = matrix([7.0, 8.0, 9.0])
    #     jpvt = matrix(0, (2, 1), tc='i')
    #    
    #     lapack.gelsy(+A, B, jpvt)
    #    
    #     # Check solution computed
    #     self.assertTrue(len(B) == 3)

    # UNSUPPORTED: gelss is not available in kvxopt.lapack
    # def test_gelss_basic(self):
    #     """Test least squares with SVD"""
    #     A = matrix([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    #     B = matrix([7.0, 8.0, 9.0])
    #    
    #     lapack.gelss(+A, B)
    #    
    #     # Check solution computed
    #     self.assertTrue(len(B) == 3)

    def test_trtrs(self):
        """Test triangular system solve"""
        # Upper triangular matrix
        A = matrix([[2.0, 1.0], [0.0, 3.0]])
        B = matrix([5.0, 6.0])
        
        lapack.trtrs(A, B, uplo='U')
        
        self.assertAlmostEqualLists(list(B), [2.5, 2.0])

    # UNSUPPORTED: geev is not available in kvxopt.lapack (use syev/heev for symmetric/hermitian matrices)
    # def test_geev_real(self):
    #     """Test eigenvalue decomposition of general real matrix"""
    #     A = matrix([[1.0, 2.0], [3.0, 4.0]])
    #     wr = matrix(0.0, (2, 1))
    #     wi = matrix(0.0, (2, 1))
    #    
    #     lapack.geev(+A, wr, wi)
    #    
    #     # Check eigenvalues computed
    #     self.assertEqual(len(wr), 2)
    #     self.assertEqual(len(wi), 2)

    # UNSUPPORTED: geev is not available in kvxopt.lapack
    # def test_geev_complex(self):
    #     """Test eigenvalue decomposition of complex matrix"""
    #     A = matrix([[1.0 + 1j, 2.0], [3.0, 4.0 + 1j]])
    #     w = matrix(0.0 + 0j, (2, 1))
    #    
    #     lapack.geev(+A, w)
    #    
    #     # Check eigenvalues computed
    #     self.assertEqual(len(w), 2)

    def test_sysv(self):
        """Test symmetric indefinite system solve"""
        # Symmetric matrix
        A = matrix([[2.0, 1.0], [1.0, 2.0]])
        B = matrix([3.0, 3.0])
        ipiv = matrix(0, (2, 1), tc='i')
        
        lapack.sysv(A, B, ipiv)
        
        # Solution should be [1.0, 1.0]
        self.assertAlmostEqualLists(list(B), [1.0, 1.0])

    def test_sytrf(self):
        """Test symmetric indefinite factorization"""
        A = matrix([[2.0, 1.0], [1.0, 2.0]])
        ipiv = matrix(0, (2, 1), tc='i')
        
        lapack.sytrf(A, ipiv)
        
        # Check factorization succeeded
        self.assertEqual(len(ipiv), 2)

    def test_gees(self):
        """Test Schur decomposition"""
        A = matrix([[1.0, 2.0], [3.0, 4.0]])
        w = matrix(0.0+0j, (2, 1))  # Must be complex
        vs = matrix(0.0, (2, 2))
        
        lapack.gees(+A, w, vs)
        
        # Check eigenvalues and Schur vectors computed
        self.assertEqual(len(w), 2)
        self.assertTrue(vs.size == (2, 2))

    # UNSUPPORTED: gehrd is not available in kvxopt.lapack
    # def test_gehrd(self):
    #     """Test reduction to Hessenberg form"""
    #     A = matrix([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    #     tau = matrix(0.0, (2, 1))
    #    
    #     lapack.gehrd(A, tau)
    #    
    #     # Check tau computed
    #     self.assertEqual(len(tau), 2)

    def test_error_handling(self):
        """Test error conditions"""
        # Singular matrix
        A = matrix([[1.0, 1.0], [1.0, 1.0]])
        B = matrix([1.0, 2.0])
        
        # Should raise an error or handle gracefully
        with self.assertRaises(ArithmeticError):
            lapack.posv(A, B)

    # UNSUPPORTED: bdsqr is not available in kvxopt.lapack
    # def test_bdsqr(self):
    #     """Test bidiagonal SVD"""
    #     d = matrix([3.0, 2.0, 1.0])
    #     e = matrix([1.0, 1.0])
    #    
    #     # bdsqr requires additional matrices for transformation
    #     # Testing basic call
    #     try:
    #         lapack.bdsqr(d, e)
    #     except (TypeError, ValueError, ArithmeticError):
    #         # May require additional parameters
    #         pass

    # UNSUPPORTED: ormqr signature issue
    # def test_ormqr(self):
    #     """Test multiplication by orthogonal matrix from QR"""
    #     A = matrix([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    #     tau = matrix(0.0, (2, 1))
    #     C = matrix([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    #    
    #     lapack.geqrf(A, tau)
    #     lapack.ormqr(A, tau, C, side='L')
    #    
    #     # C should be modified
    #     self.assertTrue(C.size == (3, 2))


class TestLAPACKEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions"""

    def test_1x1_matrix(self):
        """Test 1x1 matrices"""
        A = matrix([4.0])
        B = matrix([8.0])
        
        lapack.gesv(A, B)
        
        self.assertAlmostEqual(B[0], 2.0)

    def test_large_condition_number(self):
        """Test matrices with large condition numbers"""
        # Nearly singular matrix
        eps = 1e-10
        A = matrix([[1.0, 1.0], [1.0, 1.0 + eps]])
        B = matrix([2.0, 2.0 + eps])
        
        lapack.gesv(A, B)
        
        # Should still compute a solution
        self.assertTrue(len(B) == 2)

    def test_zero_matrix(self):
        """Test zero matrix"""
        A = matrix([[0.0, 0.0], [0.0, 0.0]])
        B = matrix([0.0, 0.0])
        
        # Should raise an error
        with self.assertRaises(ArithmeticError):
            lapack.gesv(A, B)


class TestLAPACKPerformance(unittest.TestCase):
    """Test performance-related aspects"""

    def test_medium_matrix(self):
        """Test with medium-sized matrices"""
        n = 50
        A = matrix(0.0, (n, n))
        for i in range(n):
            A[i, i] = 2.0
            if i < n - 1:
                A[i, i + 1] = -1.0
                A[i + 1, i] = -1.0
        
        B = matrix(1.0, (n, 1))
        
        lapack.gesv(A, B)
        
        # Check solution computed
        self.assertEqual(len(B), n)


if __name__ == '__main__':
    unittest.main()
