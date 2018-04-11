from linalg.common import dim,square,minus,mean,sum
from linalg.vector import argsort

class KNN_Regression:
    def __init__(self,k=3):
        self.X = None
        self.y = None
        self.k = k
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
            index = argsort(loss)[:self.k]
            ys = []
            for i in index:
                ys.append(self.y[i])
            result.append(mean(ys,axis=0))
            
        return result  