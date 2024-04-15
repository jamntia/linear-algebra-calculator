# operations.py
import numpy as np

# See if two matrices are equal
def is_equal(matrix1, matrix2):
   
    # Check each matrix has the same amount of rows and columns
    if len(matrix1) != len(matrix2) or len(matrix1[0]) != len(matrix2[0]):
        return False
    
    # Check each element of the matrices
    for i in range(len(matrix1)):
        for j in range(len(matrix1[0])):
            if matrix1[i][j] != matrix2[i][j]:
                return False
    return True

# Add two matrices together
def add(matrix1, matrix2):

    # Check if both matrices have the same number of rows and columns
    if len(matrix1) != len(matrix2) or len(matrix1[0]) != len(matrix2[0]):
        raise Exception("Error: the two matrices you entered have different dimensions")
    
    # Initialize a matrix to be returned
    fin_matrix = [[None] * len(matrix1[0]) for _ in range(len(matrix1))]

    # Update each index of the new matrix
    for i in range(len(matrix1)):
        for j in range(len(matrix1[0])):
            fin_matrix[i][j] = matrix1[i][j] + matrix2[i][j]
    
    # Return new matrix
    return fin_matrix

# Subtract one matrix from another
def subtract(matrix1, matrix2):

    # Check if both matrices have the same number of rows and columns
    if len(matrix1) != len(matrix2) or len(matrix1[0]) != len(matrix2[0]):
        raise Exception("Error: the two matrices you entered have different dimensions")
    
    # Initialize a matrix to be returned
    fin_matrix = [[None] * len(matrix1[0]) for _ in range(len(matrix1))]

    # Update each index of the new matrix
    for i in range(len(matrix1)):
        for j in range(len(matrix1[0])):
            fin_matrix[i][j] = matrix1[i][j] - matrix2[i][j]
    
    # Return new matrix
    return fin_matrix

# Get index of a matrix
def get_index(matrix, row, col):
    
    # Check if row or column is out of bounds
    if row > len(matrix) or col > len(matrix[0]) or row < 0 or col < 0:
        raise Exception("Error: Either the row or column you entered is not within the range of the matrix")
    
    # Return desired element
    return matrix[row][col]

# Scalar multipliaction
def scalar_multiplication(matrix, constant):

    # Get dimensions of the matrix
    rows = len(matrix)
    cols = len(matrix[0])

    # Make a new matrix
    result_matrix = [[None] * cols for _ in range(rows)]

    # Iterate through the matrix and multiply
    for i in range(rows):
        for j in range(cols):
            result_matrix[i][j] = matrix[i][j] * constant

    return result_matrix

# Transpose a matrix
def transpose(matrix):
   
    # Get dimensions of the matrix
    rows = len(matrix)
    cols = len(matrix[0])

    # Initialize a new matrix to store the transposed matrix
    transposed_matrix = [[None] * rows for _ in range(cols)]

    # Iterate through the original matrix and transpose its elements
    for i in range(rows):
        for j in range(cols): 
            # Assign the transposed element to the new matrix
            transposed_matrix[j][i] = matrix[i][j]

    return transposed_matrix

# Turn a matrix into rref
def rref(matrix):

    # Find number of rows and columns
    rows = len(matrix)
    cols = len(matrix[0])
    lead = 0

    # Iterate through the rows
    for r in range(rows):
        # Check if the lead column index is greater than or equal to number of clumns
        if lead >= cols:
            return matrix

        i = r

        # Move to next row if a zero entry is found
        while matrix[i][lead] == 0:
            i += 1
            if i == rows:
                i = r
                lead += 1
                if cols == lead:
                    return matrix

        # Do a row exchange
        matrix[i], matrix[r] = matrix[r], matrix[i]

        if matrix[r][lead] != 0:
            matrix[r] = [m / matrix[r][lead] for m in matrix[r]]

        
        # Row scaling
        for i in range(rows):
            if i != r:
                factor = matrix[i][lead]
                matrix[i] = [x - y * factor for x, y in zip(matrix[i], matrix[r])]

        lead += 1

    return matrix

# Create an augmented matrix
def augment_matrix(matrix, vector):
    
    # Check if vector is of length matrix
    if len(matrix) != len(vector):
        raise ValueError("Matrix rows and vector length must be equal.")

    # Update the augemented matrix
    augmented_matrix = [row + [vector[i]] for i, row in enumerate(matrix)]

    return augmented_matrix

# Gaussian Elimiation
def gaussian_elimination(matrix):
    num_rows = len(matrix)
    num_cols = len(matrix[0])
    lead = 0

    for r in range(num_rows):
        if lead >= num_cols:
            return matrix

        i = r
        while matrix[i][lead] == 0:
            i += 1
            if i == num_rows:
                i = r
                lead += 1
                if num_cols == lead:
                    return matrix

        matrix[i], matrix[r] = matrix[r], matrix[i]

        if matrix[r][lead] != 0:
            matrix[r] = [m / matrix[r][lead] for m in matrix[r]]

        for i in range(num_rows):
            if i != r:
                factor = matrix[i][lead]
                matrix[i] = [x - y * factor for x, y in zip(matrix[i], matrix[r])]

        lead += 1

    return matrix

# Check for diagonal matrix
def is_diagonal(matrix):
    num_rows = len(matrix)
    num_cols = len(matrix[0])

    # Check if the matrix is square
    if num_rows != num_cols:
        return False

    # Iterate over each element of the matrix
    for i in range(num_rows):
        for j in range(num_cols):
            # If the element is not on the main diagonal and not zero, return False
            if i != j and matrix[i][j] != 0:
                return False

    # If all elements outside the main diagonal are zero, return True
    return True

# Find the determinant of a square matrix
def determinant(matrix):
    num_rows = len(matrix)
    num_cols = len(matrix[0])

    # Check if the matrix is square
    if num_rows != num_cols:
        raise ValueError("Matrix must be square to find determinant.")

    # Base case: for 1x1 matrix, return the single element as determinant
    if num_rows == 1:
        return matrix[0][0]

    det = 0
    for j in range(num_cols):
        # Calculate the cofactor for each element in the first row
        cofactor = matrix[0][j] * ((-1) ** j) * determinant(minor(matrix, 0, j))
        det += cofactor

    return det

def minor(matrix, row, col):
    # Helper function to compute the minor of a matrix after removing a specified row and column
    return [row[:col] + row[col + 1:] for row in (matrix[:row] + matrix[row + 1:])]


# Method to calculate the inverse of a matrix
def inverse(matrix):
    num_rows = len(matrix)
    num_cols = len(matrix[0])

    # Check if the matrix is square
    if num_rows != num_cols:
        raise ValueError("Matrix must be square to find inverse.")

    # Check if determinant is nonzero
    det = determinant(matrix)
    if det == 0:
        raise ValueError("Determinant of the matrix is zero. Inverse does not exist.")

    # Initialize the identity matrix
    identity = [[1 if i == j else 0 for j in range(num_cols)] for i in range(num_rows)]

    # Keep track of the row operations performed on the identity matrix
    for i in range(num_rows):
        # Find the pivot element
        pivot = matrix[i][i]

        # Scale the current row of the identity matrix to make the pivot element 1
        for j in range(num_cols):
            identity[i][j] /= pivot

        # Perform row operations to make all elements above and below the pivot element zero
        for k in range(num_rows):
            if k != i:
                factor = matrix[k][i]
                for j in range(num_cols):
                    matrix[k][j] -= factor * matrix[i][j]
                    identity[k][j] -= factor * identity[i][j]

    return identity

# Method to find the rank of a matrix
def rank(matrix):
    # Compute the reduced row echelon form (RREF) of the matrix
    rref_matrix = rref(matrix)
    
    # Count the number of non-zero rows in the RREF
    rank = sum(1 for row in rref_matrix if any(row))
    
    return rank

# Method to find the nullity of a matrix
def nullity(matrix):
    # Compute the rank of the matrix
    matrix_rank = rank(matrix)
    
    # Calculate the nullity
    nullity = len(matrix[0]) - matrix_rank
    
    return nullity


# Method to find the null space of a matrix
def null_space(matrix):
    # Compute the reduced row echelon form (RREF) of the matrix
    rref_matrix = rref(matrix)
    
    # Determine the number of free variables (columns without leading 1's)
    num_free_variables = sum(1 for j in range(len(rref_matrix[0])) if not any(rref_matrix[i][j] for i in range(len(rref_matrix))))
    
    # Initialize a list to store the basis vectors of the null space
    null_space_basis = []
    
    # Iterate over each free variable
    for j in range(num_free_variables):
        # Create a basis vector with a 1 in the corresponding position of the free variable
        basis_vector = [0] * num_free_variables
        basis_vector[j] = 1
        
        # Back-substitute to find the values of the dependent variables
        for i in range(len(rref_matrix) - 1, -1, -1):
            dependent_var = sum(rref_matrix[i][k] * basis_vector[k] for k in range(j + 1, len(rref_matrix[0])))
            basis_vector.insert(0, -dependent_var)
        
        # Add the basis vector to the null space
        null_space_basis.append(basis_vector)
    
    return null_space_basis


# Method to find the column space of a matrix
def column_space(matrix):
    # Compute the reduced row echelon form (RREF) of the matrix
    rref_matrix = rref(matrix)
    
    # Identify the columns with leading 1's in the RREF
    leading_one_columns = []
    for j in range(len(rref_matrix[0])):
        for i in range(len(rref_matrix)):
            if rref_matrix[i][j] == 1:
                leading_one_columns.append(j)
                break
    
    # Extract the corresponding columns from the original matrix
    column_space_basis = [[matrix[i][j] for j in leading_one_columns] for i in range(len(matrix))]
    
    return column_space_basis

# Method to find the row space of a matrix
def row_space(matrix):
    # Compute the reduced row echelon form (RREF) of the matrix
    rref_matrix = rref(matrix)
    
    # Identify the rows with leading 1's in the RREF
    leading_one_rows = [i for i, row in enumerate(rref_matrix) if 1 in row]
    
    # Extract the corresponding rows from the original matrix
    row_space_basis = [matrix[i] for i in leading_one_rows]
    
    return row_space_basis