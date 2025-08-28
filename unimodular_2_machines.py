import numpy as np
from itertools import combinations, product
import sys

def create_random_arrays_for_ufp(n, m):

    a1 = [0] * (n + 1)
    a2 = [0] * (n + 1)
    
    for i in range(n + 1):
        a1[i] = np.random.randint(0, m - 1)
        # The value of a2[i] is set so that the sum with a1[i]
        # does not exceed m.
        a2[i] = np.random.randint(1, m - a1[i] + 1)
        
    return [a1, a2]


def create_ufp_matrix(n, m, a1, a2):
    """
    Given arrays a1 and a2, creates a UFP matrix of size n x m.
    """
    A = np.zeros((n, m))
    for i in range(n):
        start = a1[i]
        length = a2[i]
        if start + length > m:
            length = m - start
        A[i, start:start + length] = 1
    return A.T


def test_all_ufmp_matrices(n, m):
    """
    Tests all possible UFMP matrices of size n x m
    to determine if they are totally unimodular.
    Shows a progress bar during the process.
    """
    tu_count = 0
    not_tu_count = 0
    open('non-unimodular-ufmps.txt', 'w').close()
    # Create all possible combinations of a1 and a2 arrays
    possible_pairs_per_row = []
    for _ in range(n):
        row_pairs = []
        for a1 in range(m):
            for a2 in range(1, m - a1 + 1):
                row_pairs.append((a1, a2))
        possible_pairs_per_row.append(row_pairs)

    all_combinations = list(product(*possible_pairs_per_row))
    total_count = len(all_combinations)
    
    print(f"Start: Totally {total_count} {n}x{m} UFPM matrices will be tested...")

    # Progress bar
    for i, a_combo in enumerate(all_combinations):
        

        progress = (i + 1) / total_count
        bar_length = 50
        filled_length = int(bar_length * progress)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        
        sys.stdout.write(f'\rProgress: |{bar}| {progress:.2%} Finished.')
        sys.stdout.flush()

        # Creates UFP matrix
        a1 = [p[0] for p in a_combo]
        a2 = [p[1] for p in a_combo]
        ufp_matrix = create_ufp_matrix(n, m, a1, a2)
        
        # UFMP matrix is created from UFP matrix
        ufmp_matrix = create_ufmp_matrix(ufp_matrix)
        
        if is_totally_unimodular(ufmp_matrix):
            tu_count += 1
        else:
            not_tu_count += 1

    # Print results
    
    print("The non-unimodular Matrices:")
    with open("non-unimodular-ufmps.txt") as f:
        print(f.read())
    print("\n" + "="*50)
    print("TEST RESULTS")
    print("="*50)
    print(f"Number Of Produced Matrices: {total_count}")
    print(f"Number of Totally Unimodular (TUM) UFPM Matrices: {tu_count}")
    print(f"Number of Non-Totally Unimodular Matrices: {not_tu_count}")
    print("="*50)

def create_ufmp_matrix(M):
    # If the M matrix is not square, zero matrices must be created
    # with the correct dimensions to match the sizes.
    rows, cols = M.shape
    
    A1 = np.concatenate((M, np.zeros((rows, cols))), axis=1)

    A2 = np.concatenate((np.zeros((rows, cols)), M), axis=1)

    A3_left = np.eye(rows, cols)
    A3_right = np.eye(rows, cols)
    A3 = np.concatenate((A3_left, A3_right), axis=1)
    
    # Since the number of columns of all matrices (2*cols) is equal, concatenation is successful.
    A = np.concatenate((A1, A2, A3), axis=0)
    return A

def is_totally_unimodular(A):

    # 1. All elements of the matrix must be integers.
    if not np.allclose(A, np.round(A)):
        print("Error: All elements of the matrix are not integers.")
        return False

    m, n = A.shape
    min_dim = min(m, n)

    # 2. Test all square submatrices.
    for k in range(1, min_dim + 1):
        row_indices = list(combinations(range(m), k))
        col_indices = list(combinations(range(n), k))

        for rows in row_indices:
            for cols in col_indices:
                submatrix = A[np.ix_(rows, cols)]
                det = np.round(np.linalg.det(submatrix))
                
                if det not in [0, 1, -1]:
                    with open("non-unimodular-ufmps.txt", "a") as f:
                        f.write(f"\n"+"-" * 50)
                        f.write("\n"+f"FOUND: The matrix is not totally unimodular.")
                        f.write(f"\n"+f"Matrix is:\n{A}")
                        f.write(f"\n\n"+f"Size of the submatrix with the error: {k}x{k}")
                        f.write(f"\n"+f"Determinant value: {det}")
                        f.write(f"\n"+f"Row indices: {rows}")
                        f.write(f"\n"+f"Column indices: {cols}")
                        f.write(f"\n"+f"Submatrix with the error:\n{submatrix}")
                        f.write(f"\n"+"-" * 50)
                    return False
    
    return True

def create_ufmp_and_check(n, m):
    """
    Creates a random UFMP matrix and tests whether it is totally unimodular.
    
    Args:
        n (int): The number of rows of the UFP matrix.
        m (int): The number of columns of the UFP matrix.
    """
    a1, a2 = create_random_arrays_for_ufp(n, m)
    A = create_ufp_matrix(n, m, a1, a2)
    M = create_ufmp_matrix(A)
    if not is_totally_unimodular(A):
        print("ERROR,ERROR, ERROR, UFP matrix is not totally unimodular!")
        print("\nUFP matrix:\n", A)

    if is_totally_unimodular(M):
        print("The created UFMP matrix IS TOTALLY UNIMODULAR.")
    else:
        print("The created UFMP matrix IS NOT TOTALLY UNIMODULAR.")
        print("\nUFP matrix:\n", A)
        print("\nUFMP matrix:\n", M)
    
def test_multiple_ufmp(n, m, trials=100):
    """
    Creates a specified number of random UFMP matrices and tests
    whether each one is totally unimodular.
    
    Args:
        n (int): The number of rows of the UFP matrix.
        m (int): The number of columns of the UFP matrix.
        trials (int): The number of UFMP matrices to create.
    """
    for i in range(trials):
        print(f"\n--- Trial {i + 1} ---")
        create_ufmp_and_check(n, m)    


# Example usage:
test_all_ufmp_matrices(3, 3) #tests all matrices for small sizes


#Other Function Calls:
#test_all_ufmp_matrices(4, 4, trials=10) #tries random matrices
#test_multiple_ufmp(4, 4, trials=10)