from __future__ import print_function

from linalg.common import dim,shape,zeros
from linalg.common import mean,minus,square,shape,abs
from linalg.matrix import matrix_transpose

class Normalizer:
    def __init__(self,norm='l2'):
        assert(norm=='l1' or norm=='l2')
        self.norm=norm
        self.scaler = []
    def fit(self,X):
        assert(dim(X)==2)
        X_T = matrix_transpose(X)
        for i in range(shape(X)[1]):
            if self.norm=='l1':
                n = sum(abs(X_T[i])) 
                self.scaler.append(n)
            elif self.norm=='l2':
                n = sum(square(X_T[i])) 
                self.scaler.append(n)
            else:
                return 
    def transform(self,X):
        assert(dim(X)==2)
        R = zeros(shape(X))
        for i in range(shape(X)[0]):
            for j in range(shape(X)[1]):
                R[i][j] = X[i][j] / self.scaler[j]
        return R

    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X)
        
    def inverse_transform(self,X):
        R = zeros(shape(X))
        for i in range(shape(X)[0]):
            for j in range(shape(X)[1]):
                R[i][j] = X[i][j] * self.scaler[j]
        return R


class StandardScaler:
    pass



class MinMaxScaler:
    pass



# def main():
#     X = [[1,2,3],[4,5,6],[7,8,9]]
#     print(Normalizer().fit_transform(X))

# if __name__ == '__main__':
#     main()

