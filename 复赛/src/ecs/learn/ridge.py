
from linalg.common import (dim, mean, minus, multiply, ones, plus, reshape,
                           shape, sqrt, square, zeros)
from linalg.matrix import (hstack, identity_matrix, matrix_inverse,
                           matrix_matmul, matrix_transpose)
from linalg.common import fancy,sum


class Ridge:
    def __init__(self,alpha=1,fit_intercept=True,penalty_bias=True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.W = None
        self.dim_Y = None
        # self.bias_penalty = bias_penalty
        
        self.importance_ = None
        self.penalty_bias = penalty_bias
    def fit(self,X,y,weights=None):
        X,y = self._check(X,y)
        
        if self.fit_intercept:
            m,n = shape(X)
            bias = ones(m,1)
            X = hstack([bias,X])
        
        eye = identity_matrix(shape(X)[1])
        from linalg.matrix import diag
        if not self.penalty_bias:
            eye[0][0] = 0
        
        # add weights
        if weights!=None:
            assert(len(weights)==shape(X)[0])
            X = matrix_matmul(diag(weights),X)


        X_T = matrix_transpose(X)
        
        self.W = matrix_matmul(matrix_matmul(matrix_inverse(
            plus(matrix_matmul(X_T,X),multiply(eye,self.alpha*shape(X)[0]*2))
            # plus(matrix_matmul(X_T,X),multiply(eye,self.alpha))
        ),X_T),y)
        self.importance_ = sum(self.W,axis=1)
        if self.fit_intercept:
            self.importance_ = self.importance_[1:]

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
        if self.dim_Y==1:
            result = [x[0] for x in result]
        return result

