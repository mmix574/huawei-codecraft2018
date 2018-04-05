from linalg.common import dim, shape, reshape, ones, zeros
from linalg.common import minus,plus,sum,dot,square
from linalg.matrix import matrix_transpose, hstack
from linalg.matrix import matrix_matmul

import copy

class Lasso:
    def __init__(self, alpha=1.0, max_iter=1000, fit_intercept=True):
        self.alpha = alpha
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
        self.beta = None
        self.betas = None

    def _soft_thresholding_operator(self, x, lambda_):
        if x > 0 and lambda_ < abs(x):
            return x - lambda_
        elif x < 0 and lambda_ < abs(x):
            return x + lambda_
        else:
            return 0


    def _check(self,X,y):
        assert(shape(X)[0]==shape(y)[0])


    def _fit(self,X,y):
        self._check(X,y)
        assert(dim(y)==1)

        beta = zeros(shape(X)[1]) # row vector
        X_T = matrix_transpose(X)

        if self.fit_intercept:
            beta[0] = sum(minus(reshape(y,-1) , dot(X, beta[1:])))/(shape(X)[0])

        for _ in range(self.max_iter):
            start = 1 if self.fit_intercept else 0
            for j in range(start, len(beta)):
                tmp_beta = [x for x in beta]
                tmp_beta[j] = 0.0

                r_j = minus(reshape(y,-1) , dot(X, beta))
                # r_j = minus(reshape(y,-1) , dot(X, tmp_beta))
                arg1 = dot(X_T[j], r_j)
                arg2 = self.alpha*shape(X)[0]

                if sum(square(X_T[j]))!=0:
                    beta[j] = self._soft_thresholding_operator(arg1, arg2)/sum(square(X_T[j]))
                else:
                    beta[j] = 0

                if self.fit_intercept:
                    beta[0] = sum(minus(reshape(y,-1) , dot(X, beta[1:])))/(shape(X)[0])
        return beta


    def fit(self, X, y):
        self._check(X,y)
        if dim(y)==1:
            raw_X = X
            if self.fit_intercept:
                X = hstack([ones(shape(X)[0],1), X])

            beta = zeros(shape(X)[1]) # row vector
            X_T = matrix_transpose(X)

            if self.fit_intercept:
                beta[0] = sum(minus(reshape(y,-1) , dot(raw_X, beta[1:])))/(shape(X)[0])

            for _ in range(self.max_iter):
                print(_)
                start = 1 if self.fit_intercept else 0
                for j in range(start, len(beta)):
                    tmp_beta = [x for x in beta]
                    tmp_beta[j] = 0.0

                    r_j = minus(reshape(y,-1) , dot(X, beta))
                    # r_j = minus(reshape(y,-1) , dot(X, tmp_beta))
                    arg1 = dot(X_T[j], r_j)
                    arg2 = self.alpha*shape(X)[0]

                    if sum(square(X_T[j]))!=0:
                        beta[j] = self._soft_thresholding_operator(arg1, arg2)/sum(square(X_T[j]))
                    else:
                        beta[j] = 0

                    if self.fit_intercept:
                        beta[0] = sum(minus(reshape(y,-1) , dot(raw_X, beta[1:])))/(shape(X)[0])
                # # add whatch
                # self.beta = beta
                # self._whatch(raw_X,y)

            if self.fit_intercept:
                self.intercept_ = beta[0]
                self.coef_ = beta[1:]
            else:
                self.coef_ = beta
            self.beta = beta
            return self
        elif dim(y)==2:
            if self.fit_intercept:
                X = hstack([ones(shape(X)[0],1), X])
            y_t = matrix_transpose(y)
            betas = []
            for i in range(shape(y)[1]):
                betas.append(self._fit(X,y_t[i]))
            batas = matrix_transpose(betas)
            self.betas = batas

    def predict(self, X):
        assert(self.beta!=None or self.betas!=None)
        if self.fit_intercept:
            X = hstack([ones(shape(X)[0],1), X])
        if self.beta!=None:
            return dot(X,self.beta)
        else:
            return matrix_matmul(X,self.betas)

    def _whatch(self,X,y):
        p = self.predict(X)
        loss = sum(square(minus(p,y)))
        print(loss)