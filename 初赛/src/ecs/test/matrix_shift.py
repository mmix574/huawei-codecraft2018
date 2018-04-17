import sys
sys.path.append('..')

from linalg.matrix import shift

A= [[1,2,3],[4,5,6],[7,8,9]]

print(shift(A,1)) 

print(shift(A,0)) 

print(shift(A,-1)) 
