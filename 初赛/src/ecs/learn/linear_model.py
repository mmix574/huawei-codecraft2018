
from linalg.common import (dim, mean, minus, multiply, ones, plus, reshape,
                           shape, sqrt, square, zeros)
from linalg.matrix import (hstack, identity_matrix, matrix_inverse,
                           matrix_matmul, matrix_transpose)

# modify to fit one dimension training data
class LinearRegression:
    def __init__(self,fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.dim_Y = None
        self.W = None

    def fit(self,X,y):
        X,y = self._check(X,y)
        if self.fit_intercept:
            m,n = shape(X)
            bias = ones(m,1)
            X = hstack([bias,X])

        X_T = matrix_transpose(X)
        # print matrix_matmul(X_T,X)
        self.W = matrix_matmul(matrix_matmul(matrix_inverse(matrix_matmul(X_T,X)),X_T),y)
    
    def _check(self,X,y):
        assert((dim(X)==2 and dim(y)==2) or (dim(X)==2 and dim(y)==1))
        assert(shape(X)[0]==shape(y)[0])
        self.dim_Y = dim(y)
        if self.dim_Y == 1:
            y = [[k] for k in y]
        return X,y

    def predict(self,X):
        assert(self.W!=None)
        if self.fit_intercept:
            m,n = shape(X)
            bias = ones(m,1)
            X = hstack([bias,X])
        result = matrix_matmul(X,self.W)
        if self.dim_Y == 1:
            result = [x[0] for x in result]
        return result


