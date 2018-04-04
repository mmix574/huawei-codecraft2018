
from linalg.common import dim,shape,reshape,zeros,ones,sqrt,square
from linalg.matrix import matrix_inverse,matrix_transpose,matrix_matmul,hstack

class LinearRegression:

    def __init__(self,fit_intercept=True):
        self.fit_intercept = fit_intercept
        # self.coef_ = None 
        # self.intercept_ = None
        self.W = None

    def fit(self,X,y):
        self._check(X,y)
        if self.fit_intercept:
            m,n = shape(X)
            bias = ones(m,1)
            X = hstack([bias,X])

        X_T = matrix_transpose(X)
        # print matrix_matmul(X_T,X)
        self.W = matrix_matmul(matrix_matmul(matrix_inverse(matrix_matmul(X_T,X)),X_T),y)
    
    def _check(self,X,y):
        assert(dim(X)==2 and dim(y)==2)
        assert(shape(X)[0]==shape(y)[0])

        
    def predict(self,X):
        assert(self.W!=None)
        if self.fit_intercept:
            m,n = shape(X)
            bias = ones(m,1)
            X = hstack([bias,X])
        return matrix_matmul(X,self.W)
        
    def score(self,X,y):
        p = self.predict(X)
        p_v = reshape(p,(-1,))
        y_v = reshape(y,(-1,))
        loss = sqrt(mean(square(minus(p_v,y_v))))
        print('loss:',loss)
        
    def unfit_score(self,X,y):
        p = matrix_matmul(X,zeros((shape(X)[1],1)))
        p_v = reshape(p,(-1,))
        y_v = reshape(y,(-1,))
        loss = sqrt(mean(square(minus(p_v,y_v))))
        print('loss:',loss)


class Lasso:
    def fit(self,X,y):
        pass
    
    def predict(self,X):
        pass
    
    def score(self,X,y):
        pass

from linalg.common import dim,shape,plus,zeros,multiply,mean,sqrt,square,minus
from linalg.matrix import matrix_inverse,matrix_transpose,matrix_matmul,identity_matrix

class Ridge:

    def __init__(self,alpha=1,fit_intercept=True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.W = None

    def fit(self,X,y):
        self._check(X,y)

        if self.fit_intercept:
            m,n = shape(X)
            bias = ones(m,1)
            X = hstack([bias,X])

        X_T = matrix_transpose(X)
        self.W = matrix_matmul(matrix_matmul(matrix_inverse(
            plus(matrix_matmul(X_T,X),multiply(identity_matrix(shape(X)[1]),self.alpha))
        ),X_T),y)
    
    def _check(self,X,y):
        assert(dim(X)==2 and dim(y)==2)
        assert(shape(X)[0]==shape(y)[0])
        
    def predict(self,X):
        assert(self.W!=None)
        if self.fit_intercept:
            m,n = shape(X)
            bias = ones(m,1)
            X = hstack([bias,X])
        return matrix_matmul(X,self.W)
        
    def score(self,X,y):
        p = self.predict(X)
        p_v = reshape(p,(-1,))
        y_v = reshape(y,(-1,))
        loss = sqrt(mean(square(minus(p_v,y_v))))
        print('loss:',loss)
        
    def unfit_score(self,X,y):
        p = matrix_matmul(X,zeros((shape(X)[1],1)))
        p_v = reshape(p,(-1,))
        y_v = reshape(y,(-1,))
        loss = sqrt(mean(square(minus(p_v,y_v))))
        print('loss:',loss)

        
class ElasticNet:
    def fit(self,X,y):
        pass
    
    def predict(self,X):
        pass
    
    def score(self,X,y):
        pass