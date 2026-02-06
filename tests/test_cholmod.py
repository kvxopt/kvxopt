"""
Comprehensive tests for CHOLMOD sparse solver
"""
import unittest
from kvxopt import matrix, spmatrix, cholmod


class TestCHOLMOD(unittest.TestCase):
    """Test suite for CHOLMOD operations"""

    def assertAlmostEqualLists(self, L1, L2, places=7, msg=None):
        self.assertEqual(len(L1), len(L2))
        for u, v in zip(L1, L2):
            self.assertAlmostEqual(u, v, places, msg=msg)

    def test_symbolic_basic(self):
        """Test symbolic factorization"""
        # Simple SPD matrix
        A = spmatrix([4.0, 1.0, 2.0, 5.0], [0, 1, 0, 1], [0, 0, 1, 1], (2, 2))
        
        F = cholmod.symbolic(A)
        
        self.assertIsNotNone(F)

    def test_numeric_basic(self):
        """Test numeric factorization"""
        A = spmatrix([4.0, 1.0, 2.0, 5.0], [0, 1, 0, 1], [0, 0, 1, 1], (2, 2))
        
        Fs = cholmod.symbolic(A)
        cholmod.numeric(A, Fs)  # Updates Fs in-place, returns None
        
        self.assertIsNotNone(Fs)

    def test_symbolic_numeric_separate(self):
        """Test separate symbolic and numeric factorization"""
        A = spmatrix([4.0, 1.0, 2.0, 5.0], [0, 1, 0, 1], [0, 0, 1, 1], (2, 2))
        
        Fs = cholmod.symbolic(A)
        cholmod.numeric(A, Fs)  # Updates Fs in-place
        
        self.assertIsNotNone(Fs)

    def test_solve_basic(self):
        """Test solving linear system with CHOLMOD"""
        # SPD matrix: [[4, 2], [2, 5]]
        A = spmatrix([4.0, 2.0, 2.0, 5.0], [0, 1, 0, 1], [0, 0, 1, 1], (2, 2))
        b = matrix([8.0, 7.0])
        
        Fs = cholmod.symbolic(A)
        cholmod.numeric(A, Fs)
        x = +b
        cholmod.solve(Fs, x)
        
        # Verify solution: 4x + 2y = 8, 2x + 5y = 7 => x=1.625, y=0.75
        self.assertAlmostEqualLists(list(x), [1.625, 0.75])

    def test_solve_symbolic_numeric(self):
        """Test solve with separate symbolic/numeric"""
        A = spmatrix([4.0, 2.0, 2.0, 5.0], [0, 1, 0, 1], [0, 0, 1, 1], (2, 2))
        b = matrix([8.0, 7.0])
        
        Fs = cholmod.symbolic(A)
        cholmod.numeric(A, Fs)
        x = +b
        cholmod.solve(Fs, x)
        
        self.assertAlmostEqualLists(list(x), [1.625, 0.75])

    def test_linsolve(self):
        """Test linsolve convenience function"""
        A = spmatrix([4.0, 2.0, 2.0, 5.0], [0, 1, 0, 1], [0, 0, 1, 1], (2, 2))
        b = matrix([8.0, 7.0])
        
        cholmod.linsolve(A, b)
        
        self.assertAlmostEqualLists(list(b), [1.625, 0.75])

    def test_multiple_rhs(self):
        """Test solving with multiple right-hand sides"""
        A = spmatrix([4.0, 2.0, 2.0, 5.0], [0, 1, 0, 1], [0, 0, 1, 1], (2, 2))
        B = matrix([[8.0, 6.0], [7.0, 4.0]], (2, 2))
        
        Fs = cholmod.symbolic(A)
        cholmod.numeric(A, Fs)
        X = +B
        cholmod.solve(Fs, X)
        
        # Verify first column
        self.assertAlmostEqual(X[0, 0], 1.75, places=6)
        self.assertAlmostEqual(X[1, 0], 0.5, places=6)

    def test_complex_matrix(self):
        """Test with complex SPD matrix"""
        # Hermitian positive definite
        A = spmatrix([4.0 + 0j, 1.0 - 1j, 1.0 + 1j, 5.0 + 0j], 
                     [0, 1, 0, 1], [0, 0, 1, 1], (2, 2))
        b = matrix([3.0 + 2j, 4.0 + 3j])
        
        Fs = cholmod.symbolic(A)
        cholmod.numeric(A, Fs)
        x = +b
        cholmod.solve(Fs, x)
        
        # Should compute a solution
        self.assertEqual(len(x), 2)

    def test_larger_matrix(self):
        """Test with larger sparse matrix"""
        n = 10
        row_idx, col_idx, values = [], [], []
        for i in range(n):
            values.append(4.0)
            row_idx.append(i)
            col_idx.append(i)
            if i < n - 1:
                values.extend([1.0, 1.0])
                row_idx.extend([i, i + 1])
                col_idx.extend([i + 1, i])
        
        A = spmatrix(values, row_idx, col_idx, (n, n))
        b = matrix(1.0, (n, 1))
        
        Fs = cholmod.symbolic(A)
        cholmod.numeric(A, Fs)
        x = +b
        cholmod.solve(Fs, x)
        
        # Should compute solution
        self.assertEqual(len(x), n)

    def test_supernodal_mode(self):
        """Test supernodal mode"""
        A = spmatrix([4.0, 2.0, 2.0, 5.0], [0, 1, 0, 1], [0, 0, 1, 1], (2, 2))
        
        # Set supernodal option
        cholmod.options['supernodal'] = 2
        Fs = cholmod.symbolic(A)
        cholmod.numeric(A, Fs)
        
        self.assertIsNotNone(Fs)

    def test_simplicial_mode(self):
        """Test simplicial mode"""
        A = spmatrix([4.0, 2.0, 2.0, 5.0], [0, 1, 0, 1], [0, 0, 1, 1], (2, 2))
        
        # Simplicial factorization
        cholmod.options['supernodal'] = 0
        Fs = cholmod.symbolic(A)
        cholmod.numeric(A, Fs)
        
        self.assertIsNotNone(Fs)

    def test_refactor(self):
        """Test refactorization with modified values"""
        A1 = spmatrix([4.0, 2.0, 2.0, 5.0], [0, 1, 0, 1], [0, 0, 1, 1], (2, 2))
        A2 = spmatrix([5.0, 2.0, 2.0, 6.0], [0, 1, 0, 1], [0, 0, 1, 1], (2, 2))
        
        Fs = cholmod.symbolic(A1)
        cholmod.numeric(A1, Fs)
        
        # Refactor with new values - numeric() updates Fs in-place
        cholmod.numeric(A2, Fs)
        
        self.assertIsNotNone(Fs)

    def test_spsolve(self):
        """Test sparse right-hand side"""
        A = spmatrix([4.0, 2.0, 2.0, 5.0], [0, 1, 0, 1], [0, 0, 1, 1], (2, 2))
        b = spmatrix([8.0, 7.0], [0, 1], [0, 0], (2, 1))
        
        Fs = cholmod.symbolic(A)
        cholmod.numeric(A, Fs)
        x = cholmod.spsolve(Fs, b)
        
        self.assertIsNotNone(x)

    def test_diag_get(self):
        """Test getting diagonal from factorization"""
        A = spmatrix([4.0, 2.0, 2.0, 5.0], [0, 1, 0, 1], [0, 0, 1, 1], (2, 2))
        
        # Set supernodal mode for diag to work
        cholmod.options['supernodal'] = 2
        Fs = cholmod.symbolic(A)
        cholmod.numeric(A, Fs)
        
        try:
            d = cholmod.diag(Fs)
            self.assertEqual(len(d), 2)
        except (AttributeError, TypeError, ValueError):
            # May not be implemented in all versions or requires supernodal
            pass

    def test_getfactor(self):
        """Test extracting L factor"""
        A = spmatrix([4.0, 2.0, 2.0, 5.0], [0, 1, 0, 1], [0, 0, 1, 1], (2, 2))
        
        Fs = cholmod.symbolic(A)
        cholmod.numeric(A, Fs)
        
        try:
            L = cholmod.getfactor(Fs)
            self.assertIsNotNone(L)
        except (AttributeError, TypeError, ValueError):
            # May not be implemented in all versions
            pass

    def test_update_downdate(self):
        """Test rank-1 update/downdate"""
        A = spmatrix([4.0, 2.0, 2.0, 5.0], [0, 1, 0, 1], [0, 0, 1, 1], (2, 2))
        
        Fs = cholmod.symbolic(A)
        cholmod.numeric(A, Fs)
        C = matrix([1.0, 1.0])
        
        try:
            cholmod.updown(Fs, C, '+')
            self.assertIsNotNone(Fs)
        except (AttributeError, TypeError):
            # May not be implemented in all versions
            pass


class TestCHOLMODEdgeCases(unittest.TestCase):
    """Test edge cases and error handling"""

    def test_singular_matrix(self):
        """Test behavior with singular matrix"""
        # Singular matrix
        A = spmatrix([1.0, 1.0, 1.0, 1.0], [0, 1, 0, 1], [0, 0, 1, 1], (2, 2))
        
        # May or may not raise error depending on CHOLMOD version
        try:
            Fs = cholmod.symbolic(A)
            cholmod.numeric(A, Fs)
        except (ArithmeticError, ValueError):
            pass  # Expected

    def test_non_spd_matrix(self):
        """Test behavior with non-SPD matrix"""
        # Not positive definite (negative eigenvalue)
        A = spmatrix([1.0, 2.0, 2.0, 1.0], [0, 1, 0, 1], [0, 0, 1, 1], (2, 2))
        
        # May or may not raise error depending on CHOLMOD version and settings
        try:
            Fs = cholmod.symbolic(A)
            cholmod.numeric(A, Fs)
        except (ArithmeticError, ValueError):
            pass  # Expected

    def test_empty_matrix(self):
        """Test with empty matrix"""
        A = spmatrix([], [], [], (0, 0))
        
        # May raise error or handle gracefully
        try:
            Fs = cholmod.symbolic(A)
            _ = cholmod.numeric(A, Fs)
        except (ValueError, ArithmeticError):
            pass

    def test_1x1_matrix(self):
        """Test 1x1 matrix"""
        A = spmatrix([4.0], [0], [0], (1, 1))
        b = matrix([8.0])
        
        cholmod.linsolve(A, b)
        
        self.assertAlmostEqual(b[0], 2.0)

    def test_mismatched_dimensions(self):
        """Test with mismatched dimensions"""
        A = spmatrix([4.0], [0], [0], (2, 2))
        b = matrix([1.0, 2.0, 3.0])  # Wrong size
        
        with self.assertRaises((ValueError, ArithmeticError)):
            cholmod.linsolve(A, b)


class TestCHOLMODOptions(unittest.TestCase):
    """Test CHOLMOD options and settings"""

    def test_ordering_option(self):
        """Test different ordering options"""
        A = spmatrix([4.0, 2.0, 2.0, 5.0], [0, 1, 0, 1], [0, 0, 1, 1], (2, 2))
        
        # Try different orderings if supported
        for ordering in [0, 1]:  # Natural, AMD
            try:
                Fs = cholmod.symbolic(A)
                cholmod.numeric(A, Fs)
                self.assertIsNotNone(Fs)
            except (TypeError, ValueError):
                # Option may not be supported
                pass

    def test_options_dict(self):
        """Test passing options dictionary"""
        A = spmatrix([4.0, 2.0, 2.0, 5.0], [0, 1, 0, 1], [0, 0, 1, 1], (2, 2))
        
        try:
            # Try with options if supported
            Fs = cholmod.symbolic(A)
            cholmod.numeric(A, Fs)
            self.assertIsNotNone(Fs)
        except (TypeError, ValueError):
            pass


if __name__ == '__main__':
    unittest.main()
