from linalg.common import dim,shape
from utils import l2_loss
from utils import official_score

class BasePredictor:
    def __init__(self):
        pass
    def fit(self,X,y):
        pass
    def predict(self,X):
        if dim(X) == 1:
            return [0 for _ in X]
        R = [[0 for _ in range(shape(X)[1])]]
        for i in range(shape(X)[0]-1):
            R.append(X[i])
        return R
    def loss(self,X,y):
        y_ = self.predict(X)
        return l2_loss(y,y_)
    def score(self,X,y):
        y_ = self.predict(X)
        # print(shape(y_))
        return official_score(y,y_)