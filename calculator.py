 # Importing the operations module
from operations import *

# Example matrices
matrix1 = [[1, 2], [3, 4]]
matrix2 = [[5, 6], [7, 8]]

# Example scalar
scalar = 2

# Example vectors
vector1 = [5, 6]
vector2 = [7, 8]

# Calling functions
result_equal = is_equal(matrix1, matrix2)
result_add = add(matrix1, matrix2)
result_scalar_multiplication = scalar_multiplication(matrix1, scalar)
result_transpose = transpose(matrix1)
result_rref = rref(matrix1)
result_augment = augment_matrix(matrix1, vector1)
result_gaussian_elimination = gaussian_elimination(matrix1)
result_is_diagonal = is_diagonal(matrix1)
result_determinant = determinant(matrix1)
result_inverse = inverse(matrix1)
result_rank = rank(matrix1)
result_nullity = nullity(matrix1)
result_null_space = null_space(matrix1)
result_column_space = column_space(matrix1)
result_row_space = row_space(matrix1)

# Displaying results
print("Matrix Equality:", result_equal)
print("Matrix Addition:", result_add)
print("Scalar Multiplication:", result_scalar_multiplication)
print("Transpose:", result_transpose)
print("Reduced Row Echelon Form:", result_rref)
print("Augmented Matrix:", result_augment)
print("Gaussian Elimination:", result_gaussian_elimination)
print("Is Diagonal:", result_is_diagonal)
print("Determinant:", result_determinant)
print("Inverse:", result_inverse)
print("Rank:", result_rank)
print("Nullity:", result_nullity)
print("Null Space:", result_null_space)
print("Column Space:", result_column_space)
print("Row Space:", result_row_space)

