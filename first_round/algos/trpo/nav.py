def rotate(matrix):
    if not matrix:
        return
    n = len(matrix)

    for i in range(n):
        for j in range(i, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

    for i in range(n):
        for j in range(int(n / 2)):
            matrix[i][n - 1 - j], matrix[i][j] = matrix[i][j], matrix[i][n - 1 - j]
