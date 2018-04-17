
class dropout_estimator(BasePredictor):
    def __init__(self,estimator,parameter,drop_out=0.7):
        self.estimator = estimator
        self.parameter = parameter
        self.drop_out = drop_out

        self.shape_X = None
        self.shape_Y = None
        self.clf = None
        self.keep = None

    def fit(self,X,y,keep_hyper=False):
        assert(dim(y)==1)
        self.shape_X = shape(X)
        self.shape_Y = shape(y)

        if not keep_hyper:
            keep = [True if random.random()>self.drop_out else False for _ in range(shape(X)[1])]
            self.keep = keep
            X_ = fancy(X,-1,keep)
            clf = self.estimator(**(self.parameter))
            clf.fit(X_,y)
            self.clf = clf
        else:
            keep = self.keep
            X_ = fancy(X,-1,keep)
            clf = self.estimator(**(self.parameter))
            clf.fit(X_,y)
            self.clf = clf

    def predict(self,X):
        keep = self.keep
        X_ = fancy(X,-1,keep)
        clf = self.clf
        return clf.predict(X_)