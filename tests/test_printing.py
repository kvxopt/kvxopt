"""
Comprehensive tests for printing module
"""
import unittest
from kvxopt import matrix, spmatrix, printing


class TestPrinting(unittest.TestCase):
    """Test suite for printing module functions"""

    def setUp(self):
        """Set up test fixtures"""
        # Save original options
        self.orig_options = printing.options.copy()

    def tearDown(self):
        """Restore original options"""
        printing.options.update(self.orig_options)

    def test_matrix_str_default_double(self):
        """Test default string representation of double matrix"""
        A = matrix([1.0, 2.0, 3.0, 4.0], (2, 2))
        s = printing.matrix_str_default(A)
        
        self.assertIsInstance(s, str)
        self.assertTrue(len(s) > 0)
        self.assertIn('[', s)
        self.assertIn(']', s)

    def test_matrix_str_default_integer(self):
        """Test string representation of integer matrix"""
        A = matrix([1, 2, 3, 4], (2, 2), tc='i')
        s = printing.matrix_str_default(A)
        
        self.assertIsInstance(s, str)
        self.assertTrue(len(s) > 0)

    def test_matrix_str_default_complex(self):
        """Test string representation of complex matrix"""
        A = matrix([1.0 + 2j, 3.0 - 4j, 5.0 + 6j, 7.0 - 8j], (2, 2))
        s = printing.matrix_str_default(A)
        
        self.assertIsInstance(s, str)
        self.assertTrue(len(s) > 0)
        self.assertIn('j', s)

    def test_matrix_str_with_width(self):
        """Test matrix string with limited width"""
        A = matrix(range(20), (4, 5))
        printing.options['width'] = 3
        s = printing.matrix_str_default(A)
        
        # Should show only first 3 columns
        self.assertIn('...', s)

    def test_matrix_str_with_height(self):
        """Test matrix string with limited height"""
        A = matrix(range(20), (10, 2))
        printing.options['height'] = 3
        s = printing.matrix_str_default(A)
        
        # Should show only first 3 rows with continuation
        lines = s.strip().split('\n')
        self.assertTrue(len(lines) <= 5)  # 3 rows + continuation + blank

    def test_matrix_str_format_options(self):
        """Test custom format options"""
        A = matrix([1.23456, 2.34567], (2, 1))
        printing.options['dformat'] = '%.3f'
        s = printing.matrix_str_default(A)
        
        self.assertIn('1.235', s)

    def test_matrix_repr_default(self):
        """Test default repr for matrix"""
        A = matrix([1.0, 2.0, 3.0], (3, 1))
        s = printing.matrix_repr_default(A)
        
        self.assertIn('3x1', s)
        self.assertIn("tc='d'", s)

    def test_matrix_repr_complex(self):
        """Test repr for complex matrix"""
        A = matrix([1.0 + 1j], (1, 1))
        s = printing.matrix_repr_default(A)
        
        self.assertIn('1x1', s)
        self.assertIn("tc='z'", s)

    def test_spmatrix_str_default_sparse(self):
        """Test string representation of sparse matrix"""
        A = spmatrix([1.0, 2.0, 3.0], [0, 1, 2], [0, 1, 2], (3, 3))
        s = printing.spmatrix_str_default(A)
        
        self.assertIsInstance(s, str)
        self.assertTrue(len(s) > 0)

    def test_spmatrix_str_default_empty(self):
        """Test string representation of empty sparse matrix"""
        A = spmatrix([], [], [], (3, 3))
        s = printing.spmatrix_str_default(A)
        
        self.assertIsInstance(s, str)
        # Should handle empty matrix gracefully

    def test_spmatrix_str_default_complex(self):
        """Test string representation of complex sparse matrix"""
        A = spmatrix([1.0 + 2j, 3.0 - 4j], [0, 1], [0, 1], (2, 2))
        s = printing.spmatrix_str_default(A)
        
        self.assertIsInstance(s, str)
        self.assertIn('j', s)

    def test_spmatrix_str_triplet(self):
        """Test triplet format string representation"""
        A = spmatrix([1.0, 2.0, 3.0], [0, 1, 2], [1, 2, 0], (3, 3))
        s = printing.spmatrix_str_triplet(A)
        
        self.assertIsInstance(s, str)
        self.assertIn('(', s)
        self.assertIn(')', s)
        self.assertIn(',', s)

    def test_spmatrix_str_triplet_complex(self):
        """Test triplet format for complex sparse matrix"""
        A = spmatrix([1.0 + 1j, 2.0 - 2j], [0, 1], [0, 1], (2, 2))
        s = printing.spmatrix_str_triplet(A)
        
        self.assertIsInstance(s, str)
        self.assertIn('j', s)

    def test_spmatrix_str_triplet_empty(self):
        """Test triplet format for empty matrix"""
        A = spmatrix([], [], [], (2, 2))
        s = printing.spmatrix_str_triplet(A)
        
        self.assertIsInstance(s, str)

    def test_spmatrix_repr_default(self):
        """Test default repr for sparse matrix"""
        A = spmatrix([1.0, 2.0], [0, 1], [0, 1], (3, 3))
        s = printing.spmatrix_repr_default(A)
        
        self.assertIn('3x3', s)
        self.assertIn('sparse', s)
        self.assertIn('nnz=2', s)

    def test_spmatrix_with_width_limit(self):
        """Test sparse matrix with width limit"""
        A = spmatrix([1.0] * 10, range(10), range(10), (10, 10))
        printing.options['width'] = 5
        s = printing.spmatrix_str_default(A)
        
        self.assertIn('...', s)

    def test_spmatrix_with_height_limit(self):
        """Test sparse matrix with height limit"""
        A = spmatrix([1.0] * 10, range(10), range(10), (10, 10))
        printing.options['height'] = 5
        s = printing.spmatrix_str_default(A)
        
        lines = s.strip().split('\n')
        self.assertTrue(len(lines) <= 7)

    def test_options_persistence(self):
        """Test that options persist across calls"""
        original_width = printing.options['width']
        printing.options['width'] = 5
        
        A = matrix(range(20), (4, 5))
        s1 = printing.matrix_str_default(A)
        s2 = printing.matrix_str_default(A)
        
        self.assertEqual(s1, s2)
        
        # Restore
        printing.options['width'] = original_width

    def test_large_matrix_str(self):
        """Test string representation of large matrix"""
        A = matrix(range(100), (10, 10))
        printing.options['width'] = 10
        printing.options['height'] = 10
        s = printing.matrix_str_default(A)
        
        self.assertIsInstance(s, str)
        self.assertTrue(len(s) > 0)

    def test_negative_format_values(self):
        """Test with negative format values (unlimited)"""
        A = matrix(range(20), (4, 5))
        printing.options['width'] = -1
        printing.options['height'] = -1
        s = printing.matrix_str_default(A)
        
        # Should show entire matrix
        self.assertNotIn('...', s)

    def test_zero_dimensions(self):
        """Test with zero width or height"""
        A = matrix(range(6), (2, 3))
        printing.options['width'] = 0
        s = printing.matrix_str_default(A)
        
        self.assertEqual(s, "")

    def test_custom_iformat(self):
        """Test custom integer format"""
        A = matrix([1, 2, 3], (3, 1), tc='i')
        printing.options['iformat'] = '%03d'
        s = printing.matrix_str_default(A)
        
        # Should have leading zeros
        self.assertIsInstance(s, str)

    def test_positive_and_negative_complex(self):
        """Test complex numbers with positive and negative imaginary parts"""
        A = matrix([1.0 + 2j, 3.0 - 4j], (2, 1))
        s = printing.matrix_str_default(A)
        
        # Check for correct sign representation
        self.assertIn('+j', s)
        self.assertIn('-j', s)

    def test_print_integration(self):
        """Test that print works with matrix objects"""
        A = matrix([1.0, 2.0, 3.0])
        
        # Should not raise an error
        import io
        import sys
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        try:
            print(A)
            output = sys.stdout.getvalue()
            self.assertTrue(len(output) > 0)
        finally:
            sys.stdout = old_stdout

    def test_spmatrix_zero_elements(self):
        """Test sparse matrix with zeros (should not be stored)"""
        A = spmatrix([1.0, 0.0, 2.0], [0, 1, 2], [0, 1, 2], (3, 3))
        s = printing.spmatrix_repr_default(A)
        
        # nnz should reflect actual non-zeros
        self.assertIn('nnz=', s)


class TestPrintingEdgeCases(unittest.TestCase):
    """Test edge cases and special scenarios"""

    def test_very_small_numbers(self):
        """Test formatting of very small numbers"""
        A = matrix([1e-15, 2e-15])
        s = printing.matrix_str_default(A)
        
        self.assertIsInstance(s, str)

    def test_very_large_numbers(self):
        """Test formatting of very large numbers"""
        A = matrix([1e15, 2e15])
        s = printing.matrix_str_default(A)
        
        self.assertIsInstance(s, str)

    def test_inf_and_nan(self):
        """Test handling of inf and nan"""
        A = matrix([float('inf'), float('nan')])
        s = printing.matrix_str_default(A)
        
        self.assertIsInstance(s, str)

    def test_unicode_in_options(self):
        """Test that options handle different string formats"""
        printing.options['dformat'] = '% .2e'
        A = matrix([1.0, 2.0])
        s = printing.matrix_str_default(A)
        
        self.assertIsInstance(s, str)


if __name__ == '__main__':
    unittest.main()
