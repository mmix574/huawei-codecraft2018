from linalg.common import abs, dim, mean, minus, shape, square, sum
from linalg.matrix import diag, matrix_matmul, matrix_transpose
from linalg.vector import argsort

class KNN_Regressor:
    def __init__(self,k=3,verbose=False):
        self.X = None
        self.y = None
        self.k = k
        self.verbose = verbose

    def fit(self,X,y):
        self.X = X
        self.y = y
        
    def predict(self,X):
        result = []
        # dim_X = dim(X)

        if dim(X) == 1:
            X = [X]
        for x in X:
            loss = sum(square(minus(self.X,x)),axis=1)
            # loss = sum(abs(minus(self.X,x)),axis=1)
            index = argsort(loss)[:self.k]
            if self.verbose:
                print(index,'/len',len(loss))
            ys = []
            for i in index:
                ys.append(self.y[i])
            result.append(mean(ys,axis=0))
        return result  

# todo 
class Regularized_KNN_Regressor:
    def __init__(self,k=3,verbose=False):
        self.X = None
        self.y = None
        self.k = k
        self.verbose = verbose

    def fit(self,X,y):
        self.X = X
        self.y = y
        
    def predict(self,X):
        result = []
        # dim_X = dim(X)

        if dim(X) == 1:
            X = [X]
        for x in X:
            loss = sum(square(minus(self.X,x)),axis=1)
            # loss = sum(abs(minus(self.X,x)),axis=1)
            index = argsort(loss)[:self.k]
            if self.verbose:
                print(index,'/len',len(loss))
            ys = []
            for i in index:
                ys.append(self.y[i])
            result.append(mean(ys,axis=0))
        return result  


class Dynamic_KNN_Regressor:
    def __init__(self,k=3,verbose=False):
        self.X = None
        self.y = None
        self.k = k
        
        self.shape_X = None
        self.shape_Y = None

        self.verbose = verbose

    def fit(self,X,y):
        self.X = X
        if dim(y) == 1:
            self.y = [[k] for k in y]
        else:
            self.y = y
        self.shape_X = shape(X)
        self.shape_Y = shape(y)

    def predict(self,X):
        result = []
        # dim_X = dim(X)

        if dim(X) == 1:
            X = [X]
        for x in X:
            loss = sum(square(minus(self.X,x)),axis=1)
            
            index = argsort(loss)[:self.k]
            if self.verbose:
                print(index)


            ys = []
            for i in index:
                ys.append(self.y[i])

            k_loss_raw = sorted(loss)[:self.k]
            k_loss = [1/l if l!=0 else 0 for l in k_loss_raw]
            k_loss_sum = sum(k_loss)
            weights = [l/float(k_loss_sum) if k_loss_sum!=0 else 1 for l in k_loss]
            weight_m = diag(weights)
            ys = matrix_matmul(weight_m,ys)
            result.append(sum(ys,axis=0))


        if len(self.shape_Y)==1:
            result = matrix_transpose(result)[0]

        return result  
