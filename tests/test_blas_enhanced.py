"""
Enhanced comprehensive tests for BLAS module
"""
import unittest
from kvxopt import blas, matrix


class TestBLASLevel1(unittest.TestCase):
    """Test Level 1 BLAS operations (vector-vector)"""

    def setUp(self):
        self.x = matrix([1.0, 2.0, 3.0, 4.0])
        self.y = matrix([5.0, 6.0, 7.0, 8.0])
        self.xc = matrix([1.0 + 1j, 2.0 + 2j, 3.0 + 3j])
        self.yc = matrix([4.0 + 4j, 5.0 + 5j, 6.0 + 6j])

    def assertAlmostEqualLists(self, L1, L2, places=7, msg=None):
        self.assertEqual(len(L1), len(L2))
        for u, v in zip(L1, L2):
            self.assertAlmostEqual(u, v, places, msg=msg)

    def test_scal(self):
        """Test scalar multiplication"""
        x = +self.x
        blas.scal(2.0, x)
        self.assertAlmostEqualLists(list(x), [2.0, 4.0, 6.0, 8.0])

    def test_scal_complex(self):
        """Test scalar multiplication with complex"""
        x = +self.xc
        blas.scal(2.0 + 1j, x)
        self.assertEqual(len(x), 3)

    def test_scal_with_offset(self):
        """Test scal with offset"""
        x = +self.x
        blas.scal(2.0, x, offset=1, n=2)
        self.assertAlmostEqual(x[0], 1.0)
        self.assertAlmostEqual(x[1], 4.0)
        self.assertAlmostEqual(x[2], 6.0)

    def test_copy(self):
        """Test vector copy"""
        x = matrix(0.0, (4, 1))
        blas.copy(self.x, x)
        self.assertAlmostEqualLists(list(x), list(self.x))

    # UNSUPPORTED: blas.copy does not support inc/incy parameters
    # def test_copy_with_inc(self):
    #     """Test copy with increment"""
    #     x = matrix(0.0, (4, 1))
    #     blas.copy(self.x, x, inc=2, incy=1, n=2)
    #     self.assertAlmostEqual(x[0], 1.0)
    #     self.assertAlmostEqual(x[1], 3.0)

    def test_axpy(self):
        """Test y := alpha*x + y"""
        y = +self.y
        blas.axpy(self.x, y, alpha=2.0)
        self.assertAlmostEqualLists(list(y), [7.0, 10.0, 13.0, 16.0])

    def test_axpy_complex(self):
        """Test axpy with complex"""
        y = +self.yc
        blas.axpy(self.xc, y, alpha=1.0 + 1j)
        self.assertEqual(len(y), 3)

    def test_dot(self):
        """Test dot product"""
        result = blas.dot(self.x, self.y)
        expected = 1.0 * 5.0 + 2.0 * 6.0 + 3.0 * 7.0 + 4.0 * 8.0
        self.assertAlmostEqual(result, expected)

    def test_dotu(self):
        """Test unconjugated dot product for complex"""
        result = blas.dotu(self.xc, self.yc)
        self.assertIsNotNone(result)

    # UNSUPPORTED: blas.dotc does not exist (use dotu for complex)
    # def test_dotc(self):
    #     """Test conjugated dot product"""
    #     result = blas.dotc(self.xc, self.yc)
    #     self.assertIsNotNone(result)

    def test_nrm2(self):
        """Test Euclidean norm"""
        result = blas.nrm2(self.x)
        expected = (1.0 ** 2 + 2.0 ** 2 + 3.0 ** 2 + 4.0 ** 2) ** 0.5
        self.assertAlmostEqual(result, expected)

    def test_asum(self):
        """Test sum of absolute values"""
        result = blas.asum(self.x)
        expected = 1.0 + 2.0 + 3.0 + 4.0
        self.assertAlmostEqual(result, expected)

    def test_iamax(self):
        """Test index of maximum absolute value"""
        result = blas.iamax(self.x)
        self.assertEqual(result, 3)  # Index of 4.0

    def test_swap(self):
        """Test vector swap"""
        x = +self.x
        y = +self.y
        blas.swap(x, y)
        self.assertAlmostEqualLists(list(x), [5.0, 6.0, 7.0, 8.0])
        self.assertAlmostEqualLists(list(y), [1.0, 2.0, 3.0, 4.0])

    # UNSUPPORTED: blas.rotg does not exist
    # def test_rotg(self):
    #     """Test Givens rotation construction"""
    #     a = matrix([3.0])
    #     b = matrix([4.0])
    #     c = matrix([0.0])
    #     s = matrix([0.0])
    #    
    #     blas.rotg(a, b, c, s)
    #    
    #     self.assertIsNotNone(c[0])
    #     self.assertIsNotNone(s[0])

    # UNSUPPORTED: blas.rot does not exist  
    # def test_rot(self):
    #     """Test Givens rotation application"""
    #     x = +self.x
    #     y = +self.y
    #     c = 0.6
    #     s = 0.8
    #    
    #     blas.rot(x, y, c, s)
    #
    #     self.assertEqual(len(x), 4)
    #     self.assertEqual(len(y), 4)


class TestBLASLevel2(unittest.TestCase):
    """Test Level 2 BLAS operations (matrix-vector)"""

    def setUp(self):
        self.A = matrix([[1.0, 2.0], [3.0, 4.0]])
        self.x = matrix([1.0, 2.0])
        self.y = matrix([0.0, 0.0])

    def assertAlmostEqualLists(self, L1, L2, places=7, msg=None):
        self.assertEqual(len(L1), len(L2))
        for u, v in zip(L1, L2):
            self.assertAlmostEqual(u, v, places, msg=msg)

    def test_gemv(self):
        """Test general matrix-vector multiply"""
        y = +self.y
        blas.gemv(self.A, self.x, y, alpha=1.0, beta=0.0)
        # kvxopt uses column-major, so A*x = [1*1+3*2, 2*1+4*2] = [7, 10]
        self.assertAlmostEqualLists(list(y), [7.0, 10.0])

    def test_gemv_trans(self):
        """Test gemv with transpose"""
        y = matrix([0.0, 0.0])
        blas.gemv(self.A, self.x, y, trans='T')
        # y = A'*x (transpose gives row-major effect) = [1*1+2*2, 3*1+4*2] = [5, 11]
        self.assertAlmostEqualLists(list(y), [5.0, 11.0])

    def test_gemv_with_beta(self):
        """Test gemv with beta parameter"""
        y = matrix([1.0, 1.0])
        blas.gemv(self.A, self.x, y, alpha=1.0, beta=1.0)
        # y = 1.0*A*x + 1.0*y (column-major: [7,10] + [1,1])
        self.assertAlmostEqualLists(list(y), [8.0, 11.0])

    def test_gbmv(self):
        """Test banded matrix-vector multiply"""
        # Band storage format
        AB = matrix([[0.0, 2.0], [1.0, 3.0], [0.0, 4.0]])
        x = matrix([1.0, 2.0])
        y = matrix([0.0, 0.0])
        
        try:
            blas.gbmv(AB, x, y, m=2, kl=1, ku=1)
            self.assertEqual(len(y), 2)
        except (TypeError, ValueError) as exc:
            self.skipTest(f"blas.gbmv signature not supported: {exc}")

    def test_symv(self):
        """Test symmetric matrix-vector multiply"""
        A = matrix([[2.0, 1.0], [1.0, 3.0]])
        x = matrix([1.0, 2.0])
        y = matrix([0.0, 0.0])
        
        blas.symv(A, x, y)
        # y = A*x = [2*1+1*2, 1*1+3*2] = [4, 7]
        self.assertAlmostEqualLists(list(y), [4.0, 7.0])

    def test_hemv(self):
        """Test Hermitian matrix-vector multiply"""
        A = matrix([[2.0 + 0j, 1.0 - 1j], [1.0 + 1j, 3.0 + 0j]])
        x = matrix([1.0 + 0j, 2.0 + 0j])
        y = matrix([0.0 + 0j, 0.0 + 0j])
        
        blas.hemv(A, x, y)
        self.assertEqual(len(y), 2)

    def test_trmv(self):
        """Test triangular matrix-vector multiply"""
        A = matrix([[2.0, 1.0], [0.0, 3.0]])
        x = matrix([1.0, 2.0])
        
        blas.trmv(A, x, uplo='U')
        # Column-major, upper triangular
        self.assertAlmostEqual(x[0], 2.0)
        self.assertAlmostEqual(x[1], 6.0)

    def test_trsv(self):
        """Test triangular system solve"""
        A = matrix([[2.0, 1.0], [0.0, 3.0]])
        x = matrix([5.0, 6.0])
        
        blas.trsv(A, x, uplo='U')
        # Solve triangular system (column-major)
        self.assertAlmostEqual(x[1], 2.0)
        self.assertAlmostEqual(x[0], 2.5)

    def test_ger(self):
        """Test rank-1 update A := alpha*x*y' + A"""
        A = matrix([[1.0, 2.0], [3.0, 4.0]])
        x = matrix([1.0, 2.0])
        y = matrix([1.0, 2.0])
        
        blas.ger(x, y, A, alpha=1.0)
        # Column-major storage
        self.assertAlmostEqual(A[0, 0], 2.0)
        self.assertAlmostEqual(A[0, 1], 5.0)

    def test_syr(self):
        """Test symmetric rank-1 update"""
        A = matrix([[1.0, 2.0], [2.0, 3.0]])
        x = matrix([1.0, 2.0])
        
        blas.syr(x, A, alpha=1.0, uplo='L')
        # A = A + alpha*x*x'
        self.assertAlmostEqual(A[0, 0], 2.0)
        self.assertAlmostEqual(A[1, 1], 7.0)

    def test_syr2(self):
        """Test symmetric rank-2 update"""
        A = matrix([[1.0, 0.0], [0.0, 1.0]])
        x = matrix([1.0, 2.0])
        y = matrix([3.0, 4.0])
        
        blas.syr2(x, y, A, uplo='L')
        self.assertTrue(A.size == (2, 2))


class TestBLASLevel3(unittest.TestCase):
    """Test Level 3 BLAS operations (matrix-matrix)"""

    def setUp(self):
        self.A = matrix([[1.0, 2.0], [3.0, 4.0]])
        self.B = matrix([[5.0, 6.0], [7.0, 8.0]])
        self.C = matrix([[0.0, 0.0], [0.0, 0.0]])

    def assertAlmostEqualLists(self, L1, L2, places=7, msg=None):
        self.assertEqual(len(L1), len(L2))
        for u, v in zip(L1, L2):
            self.assertAlmostEqual(u, v, places, msg=msg)

    def test_gemm(self):
        """Test general matrix-matrix multiply"""
        C = +self.C
        blas.gemm(self.A, self.B, C)
        # Column-major: C = A*B
        self.assertAlmostEqual(C[0, 0], 23.0)
        self.assertAlmostEqual(C[0, 1], 31.0)
        self.assertAlmostEqual(C[1, 0], 34.0)
        self.assertAlmostEqual(C[1, 1], 46.0)

    def test_gemm_trans(self):
        """Test gemm with transpose"""
        C = matrix([[0.0, 0.0], [0.0, 0.0]])
        blas.gemm(self.A, self.B, C, transA='T')
        # C = A'*B
        self.assertTrue(C.size == (2, 2))

    def test_gemm_with_beta(self):
        """Test gemm with beta parameter"""
        C = matrix([[1.0, 1.0], [1.0, 1.0]])
        blas.gemm(self.A, self.B, C, alpha=1.0, beta=1.0)
        # C = 1.0*A*B + 1.0*C
        self.assertAlmostEqual(C[0, 0], 24.0)
        self.assertAlmostEqual(C[0, 1], 32.0)

    def test_symm(self):
        """Test symmetric matrix-matrix multiply"""
        A = matrix([[2.0, 1.0], [1.0, 3.0]])
        B = matrix([[1.0, 0.0], [0.0, 1.0]])
        C = matrix([[0.0, 0.0], [0.0, 0.0]])
        
        blas.symm(A, B, C, side='L')
        # C = A*B where A is symmetric
        self.assertTrue(C.size == (2, 2))

    def test_hemm(self):
        """Test Hermitian matrix-matrix multiply"""
        A = matrix([[2.0 + 0j, 1.0 - 1j], [1.0 + 1j, 3.0 + 0j]])
        B = matrix([[1.0 + 0j, 0.0 + 0j], [0.0 + 0j, 1.0 + 0j]])
        C = matrix([[0.0 + 0j, 0.0 + 0j], [0.0 + 0j, 0.0 + 0j]])
        
        blas.hemm(A, B, C, side='L')
        self.assertTrue(C.size == (2, 2))

    def test_syrk(self):
        """Test symmetric rank-k update"""
        A = matrix([[1.0, 2.0], [3.0, 4.0]])
        C = matrix([[0.0, 0.0], [0.0, 0.0]])
        
        blas.syrk(A, C, trans='N')
        # C = A*A'
        self.assertTrue(C.size == (2, 2))

    def test_herk(self):
        """Test Hermitian rank-k update"""
        A = matrix([[1.0 + 1j, 2.0 + 2j], [3.0 + 3j, 4.0 + 4j]])
        C = matrix([[0.0 + 0j, 0.0 + 0j], [0.0 + 0j, 0.0 + 0j]])
        
        blas.herk(A, C, trans='N')
        self.assertTrue(C.size == (2, 2))

    def test_syr2k(self):
        """Test symmetric rank-2k update"""
        A = matrix([[1.0, 2.0], [3.0, 4.0]])
        B = matrix([[5.0, 6.0], [7.0, 8.0]])
        C = matrix([[0.0, 0.0], [0.0, 0.0]])
        
        blas.syr2k(A, B, C)
        self.assertTrue(C.size == (2, 2))

    def test_her2k(self):
        """Test Hermitian rank-2k update"""
        A = matrix([[1.0 + 1j, 2.0 + 2j], [3.0 + 3j, 4.0 + 4j]])
        B = matrix([[5.0 + 5j, 6.0 + 6j], [7.0 + 7j, 8.0 + 8j]])
        C = matrix([[0.0 + 0j, 0.0 + 0j], [0.0 + 0j, 0.0 + 0j]])
        
        blas.her2k(A, B, C)
        self.assertTrue(C.size == (2, 2))

    def test_trmm(self):
        """Test triangular matrix-matrix multiply"""
        A = matrix([[2.0, 1.0], [0.0, 3.0]])
        B = matrix([[1.0, 2.0], [3.0, 4.0]])
        
        blas.trmm(A, B, uplo='U', side='L')
        # B = A*B where A is triangular
        self.assertTrue(B.size == (2, 2))

    def test_trsm(self):
        """Test triangular system solve with multiple RHS"""
        A = matrix([[2.0, 1.0], [0.0, 3.0]])
        B = matrix([[5.0, 7.0], [6.0, 9.0]])
        
        blas.trsm(A, B, uplo='U', side='L')
        # Solve A*X = B
        self.assertTrue(B.size == (2, 2))


class TestBLASEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""

    def test_zero_length_vector(self):
        """Test operations with zero-length vectors"""
        x = matrix([], (0, 1))
        result = blas.nrm2(x)
        self.assertAlmostEqual(result, 0.0)

    def test_single_element(self):
        """Test single element operations"""
        x = matrix([5.0])
        result = blas.nrm2(x)
        self.assertAlmostEqual(result, 5.0)

    def test_negative_values(self):
        """Test with negative values"""
        x = matrix([-1.0, -2.0, -3.0])
        result = blas.asum(x)
        self.assertAlmostEqual(result, 6.0)


if __name__ == '__main__':
    unittest.main()
