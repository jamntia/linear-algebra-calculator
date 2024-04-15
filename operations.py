# operations.py

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