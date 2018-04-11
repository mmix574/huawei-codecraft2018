import sys
sys.path.append('..')

from preprocessing import exponential_smoothing

A = [[1,2,3],[4,5,6],[7,8,9],[0,0,0],[1,2,3]]

print(A)
print(exponential_smoothing(A,alpha=0.5))