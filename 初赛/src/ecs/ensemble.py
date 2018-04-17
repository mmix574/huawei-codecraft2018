import random

from linalg.common import multiply, plus, shape
from metrics import l2_loss, official_score


class bagging_estimator:
    def __init__(self,estimator,parameter,max_clf = 100):
        self.estimator = estimator
        self.parameter = parameter
        self.max_clf = max_clf
        self.clfs = []

    def bagging(self,X_train,Y_train):
        N = shape(X_train)[0]
        index = []
        for i in range(N):
            index.append(random.randrange(N))
        X = [X_train[i] for i in index]
        y = [Y_train[i] for i in index]
        return X,y
    
    def fit(self,X,y):
        for i in range(self.max_clf):
            clf = self.estimator(**self.parameter)
            _X,_y = self.bagging(X,y)
            clf.fit(_X,y)
            self.clfs.append(clf)
        
    def predict(self,X):
        prediction = None
        for i in range(self.max_clf):
            if not prediction:
                prediction = self.clfs[i].predict(X)
            else:
                prediction = plus(prediction,self.clfs[i].predict(X))
        prediction = multiply(prediction,1/float(self.max_clf))
        return prediction


# data selection based on model
def bagging_with_model(regressor_instance,X_train,Y_train,X_val,Y_val,bagging_size=None,max_iter=100,verbose=False,scoring='score'):
    def bagging(X_train,Y_train,bagging_size=None):
        N = shape(X_train)[0]
        if bagging_size!=None:
            N = bagging_size
        index = []
        for i in range(N):
            index.append(random.randrange(N))
        X = [X_train[i] for i in index]
        y = [Y_train[i] for i in index]
        return X,y
    assert(scoring=='loss' or scoring=='score')
    if scoring=='score':
        max_score = None
        best_XY = None
        for i in range(max_iter):
            X,y = bagging(X_train,Y_train,bagging_size=bagging_size)
            regressor_instance.fit(X,y)
            p = regressor_instance.predict(X_val)
            score = official_score(p,Y_val)
            # score = regressor_instance.score(X_val,Y_val)
            if not max_score or score>max_score:
                if verbose:
                    print(score)
                score =  max_score
                best_XY = (X,y)
        
        X_train,Y_train = best_XY
        return X_train,Y_train
    elif scoring=='loss':
        min_loss = None
        best_XY = None
        for i in range(max_iter):
            X,y = bagging(X_train,Y_train,bagging_size=bagging_size)
            regressor_instance.fit(X,y)
            # loss = regressor_instance.loss(X_val,Y_val)
            p = regressor_instance.predict(X_val)
            loss = l2_loss(p,Y_val)

            if not min_loss or loss<min_loss:
                if verbose:
                    print(loss)
                min_loss =  loss
                best_XY = (X,y)
        X_train,Y_train = best_XY
        return X_train,Y_train
