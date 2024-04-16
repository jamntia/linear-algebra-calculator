import unittest
from operations import *

class TestOperations(unittest.TestCase):
    def test_is_equal(self):
        matrix1 = [[1, 2], [3, 4]]
        matrix2 = [[1, 2], [3, 4]]
        matrix3 = [[1, 2], [3, 5]]
        self.assertTrue(is_equal(matrix1, matrix2))
        self.assertFalse(is_equal(matrix1, matrix3))
    
    def test_add(self):
        matrix1 = [[1, 2], [3, 4]]
        matrix2 = [[5, 6], [7, 8]]
        result = [[6, 8], [10, 12]]
        self.assertEqual(add(matrix1, matrix2), result)
   
    def test_subtract(self):
        matrix1 = [[5, 6], [7, 8]]
        matrix2 = [[1, 2], [3, 4]]
        result = [[4, 4], [4, 4]]
        self.assertEqual(subtract(matrix1, matrix2), result)

    def test_get_index(self):
        matrix = [[1, 2], [3, 4]]
        self.assertEqual(get_index(matrix, 0, 1), 2)
        self.assertEqual(get_index(matrix, 1, 1), 4)

    def test_scalar_multiplication(self):
        matrix = [[1, 2], [3, 4]]
        scalar = 2
        result = [[2, 4], [6, 8]]
        self.assertEqual(scalar_multiplication(matrix, scalar), result)

    def test_transpose(self):
        matrix = [[1, 2], [3, 4]]
        result = [[1, 3], [2, 4]]
        self.assertEqual(transpose(matrix), result)
    
    def test_rref(self):
        matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        result = [[1, 0, -1], [0, 1, 2], [0, 0, 0]]
        self.assertEqual(rref(matrix), result)

    def test_augment_matrix(self):
        matrix = [[1, 2], [3, 4]]
        vector = [5, 6]
        result = [[1, 2, 5], [3, 4, 6]]
        self.assertEqual(augment_matrix(matrix, vector), result)

    def test_gaussian_elimination(self):
        matrix = [[2, 1, -1, 8], [-3, -1, 2, -11], [-2, 1, 2, -3]]
        result = [[1, 0, 0, 2], [0, 1, 0, 3], [0, 0, 1, -1]]
        self.assertEqual(gaussian_elimination(matrix), result)

    def test_is_diagonal(self):
        diagonal_matrix = [[1, 0, 0], [0, 2, 0], [0, 0, 3]]
        non_diagonal_matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        self.assertTrue(is_diagonal(diagonal_matrix))
        self.assertFalse(is_diagonal(non_diagonal_matrix))
    def test_determinant(self):
        square_matrix = [[1, 2], [3, 4]]
        self.assertEqual(determinant(square_matrix), -2)

    def test_inverse(self):
        invertible_matrix = [[1, 2], [3, 4]]
        non_invertible_matrix = [[1, 2], [2, 4]]
        inverse_result = [[-2, 1], [1.5, -0.5]]
        with self.assertRaises(ValueError):
            inverse(non_invertible_matrix)
        self.assertEqual(inverse(invertible_matrix), inverse_result)

    def test_rank(self):
        full_rank_matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        rank_deficient_matrix = [[1, 2], [3, 6]]
        self.assertEqual(rank(full_rank_matrix), 2)
        self.assertEqual(rank(rank_deficient_matrix), 1)

    def test_nullity(self):
        full_rank_matrix = [[1, 0], [0, 1]]
        rank_deficient_matrix = [[1, 2, 3], [4, 8, 12]]  
        self.assertEqual(nullity(full_rank_matrix), 0)
        self.assertEqual(nullity(rank_deficient_matrix), 1)
    
    def test_column_space(self):
        matrix = [[1, 2], [3, 6], [2, 4]]
        column_space_basis = [[1, 3, 2], [2, 6, 4]] 
        self.assertEqual(column_space(matrix), column_space_basis)

    def test_null_space(self):
        matrix1 = [[1, 2], [2, 4]]
        matrix2 = [[1, 2, 3], [3, 6, 9]]
        null_space_basis1 = [[-2, 1]] 
        null_space_basis2 = [[-2, 1, 0]] 
        self.assertEqual(null_space(matrix1), null_space_basis1)
        self.assertEqual(null_space(matrix2), null_space_basis2)

    def test_row_space(self):
        matrix = [[1, 2], [3, 6], [2, 4]]
        row_space_basis = [[1, 2], [0, 0], [0, 0]]
        self.assertEqual(row_space(matrix), row_space_basis)
    
    def test_matrix_multiplication(self):
        matrix1 = [[1, 2], [3, 4]]
        matrix2 = [[5, 6], [7, 8]]
        result = [[19, 22], [43, 50]]
        self.assertEqual(matrix_multiplication(matrix1, matrix2), result)
if __name__ == '__main__':
    unittest.main()