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