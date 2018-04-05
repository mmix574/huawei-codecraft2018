from __future__ import print_function

from linalg.common import dim,shape,zeros,multiply
from linalg.common import mean,minus,square,shape,abs,sqrt
from linalg.matrix import matrix_transpose,matrix_copy



def stdev(X):
    # X = matrix_copy(X)
    X_T = matrix_transpose(X)
    m = mean(X,axis=1)
    R = []
    for j in range(shape(X)[1]):
        R.append(sqrt(mean(square(minus(X_T[j],m[j])))))
    return R

# column vector normalize -- axix=1
# fix bug
def normalize(X,norm='l2',axis=1,return_norm=False):
    assert(axis==0 or axis==1)
    X_T = matrix_transpose(X)

    norm = []
    if axis == 0:
        A = matrix_copy(X)

        for i in range(shape(X)[0]):
            n = sqrt(sum(square(X[i])))
            A[i] = (multiply(X[i],1/float(n)))
            norm.append(n)
    elif axis == 1:
        A = matrix_transpose(X)
    
        for j in range(shape(X)[1]):
            n = sqrt(sum(square(X_T[j])))
            A[j] = (multiply(X_T[j],1/float(n)))
            norm.append(n)
        
        A = matrix_transpose(A)

    if return_norm:
        return A,norm
    else:
        return A

def minmax():
    pass

def standardizing():
    pass