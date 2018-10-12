from __future__ import division, print_function
import numpy as np
from collections import Counter

# TeacherNet weights
W1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
W2 = np.array([[13, 14, 15], [16, 17, 18], [19, 20, 21]])

# Matrix dimensions
m, n = W1.shape
_, p = W2.shape
q = n + 10

# Random mapping function
g = {j: j if j < n else np.random.choice(n) for j in range(q)}
# g = {0: 0, 1: 1, 2: 2, 3: 2, 4: 0}

# StudentNet weights
U1 = np.array([[W1[x][y] if y < n else W1[x][g[y]] for y in range(q)] for x in range(m)])
U2 = np.array([[W2[x][y] / Counter(g.values())[x] if x < n else W2[g[x]][y] / Counter(g.values())[g[x]] for y in range(p)] for x in range(q)])

x1 = np.array([1, 2, 3, 4])
x2 = np.array([5, 6, 7, 8])
x3 = np.array([-1, 0, 3, -18])

print("TeacherNet: ")
print(x1.dot(W1).dot(W2))
print("StudentNet: ")
print(x1.dot(U1).dot(U2))

assert np.isclose(x1.dot(W1).dot(W2), x1.dot(U1).dot(U2)).all()
assert np.isclose(x2.dot(W1).dot(W2), x2.dot(U1).dot(U2)).all()
assert np.isclose(x3.dot(W1).dot(W2), x3.dot(U1).dot(U2)).all()

