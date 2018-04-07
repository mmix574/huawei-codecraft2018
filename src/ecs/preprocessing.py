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
    assert(norm=='l2' or norm=='l1')
    X_T = matrix_transpose(X)

    norms = []
    if axis == 0:
        A = matrix_copy(X)

        for i in range(shape(X)[0]):
            n = 0
            if norm=='l2':
                n = sqrt(sum(square(X_T[j])))
            elif norm=='l1':
                n = sqrt(sum(abs(X_T[j])))
            if n!=0:
                A[i] = (multiply(X[i],1/float(n)))
            norms.append(n)
    elif axis == 1:
        A = matrix_transpose(X)
        for j in range(shape(X)[1]):
            n = 0
            if norm=='l2':
                n = sum(square(X_T[j]))
            elif norm=='l1':
                n = sum(abs(X_T[j]))
            if n!=0:
                A[j] = (multiply(X_T[j],1/float(n)))
            norms.append(n)
        
        A = matrix_transpose(A)

    if return_norm:
        return A,norms
    else:
        return A

def minmax():
    pass

def standardizing():
    pass

class Random_Feature:
    def fit(self):
        pass
    
    def fit_transform(self):
        pass
    
    def transform(self):
        pass

    def inverse_transform(self):
        pass