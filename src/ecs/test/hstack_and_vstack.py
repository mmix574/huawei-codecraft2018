import sys
sys.path.append('..')


from linalg.common import shape
from linalg.matrix import hstack,vstack,shift

A = [[0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 4.0, 3.0, 0.0, 4.0, 8.0, 0.0, 0.0, 4.0], [0.0, 0.0, 3.0, 2.0, 14.0, 1.0, 1.0, 6.0, 3.0, 0.0, 2.0, 14.0, 0.0, 27.0, 21.0], [0.0, 2.0, 1.0, 1.0, 4.0, 3.0, 0.0, 7.0, 7.0, 0.0, 7.0, 5.0, 0.0, 1.0, 5.0], [0.0, 0.0, 1.0, 0.0, 3.0, 1.0, 0.0, 4.0, 0.0, 0.0, 3.0, 1.0, 0.0, 8.0, 0.0], [0.0, 2.0, 3.0, 0.0, 4.0, 8.0, 1.0, 14.0, 1.0, 0.0, 14.0, 0.0, 0.0, 4.0, 0.0], [2.0, 4.0, 1.0, 1.0, 4.0, 0.0, 4.0, 12.0, 6.0, 0.0, 3.0, 9.0, 1.0, 4.0, 1.0], [1.0, 2.0, 1.0, 0.0, 3.0, 3.0, 11.0, 28.0, 8.0, 4.0, 4.0, 1.0, 0.0, 0.0, 0.0]]


# print(shift(A,1))
print(shape(vstack([A])))
print(shape(vstack([A,A,A])))
print(vstack([A,A,A]))


print(shape(hstack([A])))
print(shape(hstack([A,shift(A,1),A])))
print(hstack([A,shift(A,1),A]))


