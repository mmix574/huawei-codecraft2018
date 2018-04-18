from .common import dim,shape
import math
from .common import plus
# ---------------------vector funciton----------------------

# return element counts of none zero
def count_nonezero(A):
    c = 0
    for i in A:
        if i==True or i!=0:
            c += 1
    return c

def arange(a,b,count):
    h = (b - a)/float(count)
    return [i*h+a for i in range(count)]


# add @2018-04-08
def argsort(X):
    assert(dim(X)==1)
    return sorted(range(len(X)), key=X.__getitem__)

# add @2018-04-16
def argmin(A):
    assert(dim(A)==1)
    min_index = 0
    for i in range(len(A)):
        if A[i] < A[min_index]:
            min_index = i
    return min_index

# add @2018-04-18
def argmax(A):
    assert(dim(A)==1)
    min_index = 0
    for i in range(len(A)):
        if A[i] > A[min_index]:
            min_index = i
    return min_index