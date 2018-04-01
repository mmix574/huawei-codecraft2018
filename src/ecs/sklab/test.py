from load_data import load_xy
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler,Normalizer,MinMaxScaler



N = 6

X_train,Y_train,X_val,Y_val = load_xy(prefix='../data/',argumentation = True,N=N)





from sklearn.linear_model import Lasso
from sklearn.cross_validation import cross_val_score
clf = LinearRegression()

def sample_X(X,i):
    K = X.reshape(X.shape[0],-1,20)
    return K[:,:,i:i+1].reshape(K.shape[0],-1)
def sample_Y(Y,i):
    return Y[:,i]


sds = StandardScaler()



# for i in range(Y_train.shape[1]):
#     # print(sample_X(X_train,i))
#     # print(sample_Y(Y_train,i))
#     scores = cross_val_score(clf,sample_X(X_train,i),sample_Y(Y_train,i),cv=10)
#     print(scores)
#     # clf.fit(sample_X(X_train,i),sample_Y(Y_train,i))
#     # fs = clf.score(sample_X(X_train,i),sample_Y(Y_train,i))
#     # vs = clf.score(sample_X(X_val,i),sample_Y(Y_val,i))
#     # print(fs,vs)


# print(
# sds.fit_transform(sample_X(X_train,0))
# )

import numpy as np

from sklearn.preprocessing import normalize

A = np.arange(9).reshape(3,3)

# B = np.arange(9).reshape(3,3)
# print(Normalizer(norm='l2').fit_transform(A,axis=1)
# )

A_transform,norm = normalize(A,axis=0,norm='l2',return_norm=True)


print(A_transform*norm)


