import sys 
sys.path.append('..')

import matplotlib.pyplot as plt
import math

original_x = [1,2,3,4,5,6,7,8,9]
original_y = [math.sin(k) for k in original_x]

from linalg.matrix import matrix_inverse,matrix_transpose,matrix_mutmul
from linalg.vector import arange

print(original_x)
print(original_y)

family = [lambda x:1,lambda x:x,lambda x:x**2,lambda x:math.pow(x,3),lambda x:math.pow(x,4)]
X = [[f(k) for f in family] for k in original_x]
Y = [[k] for k in original_y]
X_T = matrix_transpose(X)


b = matrix_mutmul(matrix_mutmul(matrix_inverse(matrix_mutmul(X_T,X)),X_T),Y)


xx = arange(original_x[0],original_x[-1],100)
yy = [] 
for k in xx:
    yy.append(sum([b[i][0]*family[i](k) for i in range(len(family))]))
plt.plot(xx,yy,label='fitting curve')
plt.scatter(original_x,original_y,label='original data',c='red')
plt.legend()
plt.show()