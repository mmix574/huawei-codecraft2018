from __future__ import print_function

from linalg.common import (abs, dim, fancy, mean, minus, multiply, plus, shape,
                           sqrt, square, zeros)
from linalg.matrix import matrix_copy, matrix_transpose


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
def normalize(X,norm='l2',axis=1,return_norm=False,return_norm_inv=False):
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

    norms_inv = [0 if x==0 else 1/float(x)for x in norms]
    if return_norm and return_norm_inv:
        return A,norms,norms_inv
    elif return_norm:
        return A,norms
    elif return_norm_inv:
        return A,norms_inv
    else:
        return A

def minmax_scaling(X,axis=1):
    assert(axis==1)
    R = []
    for j in range(shape(X)[1]):
        col = fancy(X,None,j)     
        max_ = max(col)
        min_ = min(col)
        mean_ = mean(col)
        if max_ - min_==0:
            R.append(col)
        else:
            R.append([(x-mean_)/(max_-min_) for x in col])    
    return matrix_transpose(R)
    
def standard_scaling(X,axis=1):
    R = []
    for j in range(shape(X)[1]):
        col = fancy(X,None,j)     
        mean_ = mean(col)
        std = sqrt(mean(square(minus(col,mean_))))
        if std==0:
            R.append(col)
        else:
            R.append([(x-mean_)/std for x in col])    
    return matrix_transpose(R)

def maxabs_scaling(X,axis=1):
    assert(axis==1)
    R = []
    for j in range(shape(X)[1]):
        col = fancy(X,None,j)     
        max_ = max(abs(col))
        mean_ = mean(col)
        if max_ == 0:
            R.append(col)
        else:
            R.append([(x-mean_)/(max_) for x in col])
                
    return matrix_transpose(R)

# 
# def weight_decay_smoothing(X,axis=0,weight=0.5):
#     assert(axis==0)

def exponential_smoothing(A,axis=0,alpha=0.1):
    assert(axis==0)
    R = []
    C = zeros(shape(A)[1])
    for i in range(shape(A)[0]):
        P = multiply(A[i],(1-alpha))
        Q = multiply(C,alpha)
        C = plus(P,Q)
        R.append(C)
    return R
