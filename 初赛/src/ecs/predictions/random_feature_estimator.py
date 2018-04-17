import math
import random
from linalg.common import dim, fancy, shape

from .base import BasePredictor

from ..model_selection import cross_val_score

class random_feature_estimator(BasePredictor):
    def __init__(self,estimator,parameter,max_iter=20,drop_out=0.3,scoring='score'):
        self.estimator = estimator
        self.parameter = parameter
        self.max_iter = max_iter
        self.drop_out = drop_out
        self.scoring = scoring

        self.keeps = []
        self.clfs = []
        self.shape_Y = None

    def _rand_X(self,X):
        N = shape(X)[1]
        keep_length = math.ceil((1-self.drop_out)*N)
        keep_set = set()
        while len(keep_set)!=keep_length:
            i = random.randrange(N)
            if i not in keep_set:
                keep_set.add(i)
        keep = [True if i in keep_set else False for i in range(N)]
        X_ = fancy(X,-1,keep)
        return X_,keep

    def _get_keep_X(self,X,keep):
        return fancy(X,-1,keep)

    # implement @ 2018-04-08
    # cross_val_score
    def train_cv(self,X,y,shuffle=False,cv='full'):
        assert(type(cv)==int or cv=='full')
        assert(dim(X)==2 and dim(y)==2)
        self.shape_Y = shape(y)
        for i in range(shape(y)[1]):
            max_score = None
            best_clf = None
            best_keep = None
            y_ = fancy(y,-1,i)
            for _ in range(self.max_iter):
                clf = self.estimator(**(self.parameter))
                X_,keep = self._rand_X(X)
                clf.fit(X_,y_)
                score = cross_val_score(clf,X,y,return_mean=True,cv=cv,shuffle=shuffle) 
                if not max_score or max_score<score:
                    max_score = score
                    best_clf = clf
                    best_keep = keep
            self.keeps.append(best_keep)
            self.clfs.append(best_clf) 

    def train(self,X,y,X_val,Y_val):
        assert(dim(X)==2 and dim(y)==2)
        self.shape_Y = shape(y)

        for i in range(shape(y)[1]):
            max_score = None
            best_clf = None
            best_keep = None
            y_ = fancy(y,-1,i)
            for _ in range(self.max_iter):
                clf = self.estimator(**(self.parameter))
                X_,keep = self._rand_X(X)
                clf.fit(X_,y_)
                score = clf.score(self._get_keep_X(X_val,keep),fancy(Y_val,-1,i))

                if not max_score or max_score<score:
                    max_score = score
                    best_clf = clf
                    best_keep = keep

            self.keeps.append(best_keep)
            self.clfs.append(best_clf) 

    def retrain(self,X,y):
        assert(len(self.keeps)!=0)
        for i in range(self.shape_Y[1]):
            X_ = self._get_keep_X(X,self.keeps[i])
            self.clfs[i].fit(X_,fancy(y,-1,i))

    def predict(self,X):
        assert(dim(X)==2)
        result = []
        for i in range(self.shape_Y[1]):
            X_ = self._get_keep_X(X,self.keeps[i])
            result.append(self.clfs[i].predict(X_))
        return matrix_transpose(result)
