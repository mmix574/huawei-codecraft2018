from .common import dim,shape
import math
# ---------------------vector funciton----------------------

# return element counts of none zero
def count_nonezero(A):
    c = 0
    for i in A:
        if i==True or i!=0:
            c += 1
    return c

def arange(a,b,count):
    h = (b - a)/count
    return [i*h+a for i in range(count)]

