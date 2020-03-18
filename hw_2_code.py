import numpy as np
import numpy.linalg as nl

######Problem 1
###(a)
A = np.array([[3,-1,1],[-1,2,-1],[1,-1,4]])
print(A)

###(b)
eigensystem = nl.eig(A)
print(eigensystem)

######Problem 2
###(a)
L = nl.cholesky(A)
print(L)

###(b)
u = np.random.normal(size=(3,100000))
cov_u = np.cov(u)
print(cov_u)

###(c)
v = np.dot(L, u)
cov_v = np.cov(v)
print(cov_v)