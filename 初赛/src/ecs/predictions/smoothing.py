# add @2018-03-28
class Smoothing(BasePredictor):
    def __init__(self,weight_decay=0.4):
        BasePredictor.__init__(self)
        self.weight_decay = weight_decay
        self.shape_X = None
        self.shape_Y = None

    def fit(self,X,y):
        self.shape_X = shape(X)
        self.shape_Y = shape(y)

    def predict(self,X):
        X = reshape(X,(shape(X)[0],-1,self.shape_Y[-1]))
        X = X[::-1]
        N = shape(X)[1]

        norm = sum([math.pow(self.weight_decay,k) for k in range(N)])
        W = [math.pow(self.weight_decay,k)/norm for k in range(N)]
        W = [W for _ in range(self.shape_Y[-1])]
        W = matrix_transpose(W)
        R = []
        for i in range(shape(X)[0]):
            R.append(sum(multiply(X[i],W),axis=0))
        return R