import math
import random

from linalg.common import apply, shape, zeros,dim,minus,mean,sum,sqrt,square
from linalg.common import flatten

# reinforcement learning 
# no feature eningneering,or at lease automatica feature eningeering

# give confidence of features
# with new sample 

from linalg.matrix import matrix_matmul

class rl:
    def __init__(self,max_iter=1000,sparse=10,family=[lambda x:1,lambda x:x,lambda x:math.pow(x,2)]):
        self.max_iter = max_iter        
        self.family = family
        self.importance_ = None

    def random_w(self,s):
        assert(len(s)==2)
        R = zeros(s)
        for i in range(shape(R)[0]):
            for j in range(shape(R)[1]):
                R[i][j] = random.random()
        return R
    
    def fit(self,X,y):
        assert(dim(X)==2)
        assert(dim(y)==1 or dim(y)==2)
        self.shape_X = shape(X)
        self.shape_Y = shape(y)

        if dim(y) == 1:
            y = [[k] for k in y]
        
        best_w = None
        min_err = None
        for i in range(self.max_iter):
            
            

            W = self.random_w((shape(X)[1],shape(y)[1]))
            
            y_ = matrix_matmul(X,W)
            err = mean(sqrt(mean(square(minus(y,y_)),axis=1)))
            if not best_w or min_err>err:
                best_w = W
                min_err = err
            print(err)
        self.W = best_w

    def predict(self,X):
        if len(self.shape_Y) == 2:
            return matrix_matmul(X,self.W)
        else:
            return flatten(matrix_matmul(X,self.W))
        

    