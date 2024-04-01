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