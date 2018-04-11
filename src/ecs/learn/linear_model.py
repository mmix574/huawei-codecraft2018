
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

    # # !! prepare to remove 
    # def score(self,X,y,scoring='mse'):
    #     assert(scoring=='mse')
    #     p = self.predict(X)
    #     p_v = reshape(p,(-1,))
    #     y_v = reshape(y,(-1,))
    #     loss = sqrt(mean(square(minus(p_v,y_v))))
    #     print('loss:',loss)


class Ridge:
    def __init__(self,alpha=1,fit_intercept=True,bias_no_penalty=False):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.W = None
        self.dim_Y = None
        self.bias_no_penalty = bias_no_penalty

    def fit(self,X,y):
        X,y = self._check(X,y)
        
        if self.fit_intercept:
            m,n = shape(X)
            bias = ones(m,1)
            X = hstack([bias,X])
        
        eye = identity_matrix(shape(X)[1])
        if self.bias_no_penalty==True:
            eye[0][0] = 0
       
        X_T = matrix_transpose(X)
        self.W = matrix_matmul(matrix_matmul(matrix_inverse(
            plus(matrix_matmul(X_T,X),multiply(eye,self.alpha))
        ),X_T),y)
    
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

    # # !! prepare to remove 
    # def score(self,X,y,scoring='mse'):
    #     assert(scoring=='mse')
    #     p = self.predict(X)
    #     p_v = reshape(p,(-1,))
    #     y_v = reshape(y,(-1,))
    #     loss = sqrt(mean(square(minus(p_v,y_v))))
    #     print('loss:',loss)
