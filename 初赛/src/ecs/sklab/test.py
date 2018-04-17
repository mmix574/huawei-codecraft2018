from load_data import load_xy
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler,Normalizer,MinMaxScaler



N = 4

X_train,Y_train,X_val,Y_val = load_xy(prefix='../data/',argumentation = True,N=N)


from sklearn.linear_model import Lasso
from sklearn.cross_validation import cross_val_score
from sklearn.preprocessing import normalize

def sample_X(X,i):
    K = X.reshape(X.shape[0],-1,20)
    return K[:,:,i:i+1].reshape(K.shape[0],-1)
def sample_Y(Y,i):
    return Y[:,i]




import numpy as np

# X_train,X_norm = normalize(X_train,axis=0,return_norm=True,norm='l1')
# Y_train,Y_norm = normalize(Y_train,axis=0,return_norm=True)

ss = MinMaxScaler()
X_train = ss.fit_transform(X_train)
X_val = ss.transform(X_val)


# X_val/=X_norm
# Y_val/=Y_norm

def score(Y,Y_):
    Y = np.array(Y)
    Y_ = np.array(Y_)
    if(len(np.shape(Y))==1):
        Y = Y.reshape(-1,1)
    if(len(np.shape(Y_))==1):
        Y_ = Y_.reshape(-1,1)

    A = np.power(np.sum((Y-Y_)**2,axis=1),0.5)
    B = np.power(np.sum((Y)**2,axis=1),0.5) + np.power(np.sum((Y_)**2,axis=1),0.5)
    # print(1-np.array(A/B))
    score = 1-np.mean(np.array(A/B))
    # print('-->',1-np.mean(np.array(A/B)))
    print(score)
    return score


from sklearn.linear_model import Lasso,Ridge
# clf = Lasso(fit_intercept=False)
clf = Ridge(fit_intercept=False)


for i in range(Y_train.shape[1]):
    clf.fit(sample_X(X_train,i),sample_Y(Y_train,i))
    print(clf.score(sample_X(X_val,i),sample_Y(Y_val,i)))
    # clf.fit(np.log(sample_X(X_train,i)+1),sample_Y(Y_train,i))
    # print(clf.score(np.log(sample_X(X_val,i)+1),sample_Y(Y_val,i)))

    # clf.fit(X_train,sample_Y(Y_train,i))
    # print(clf.score(X_val,sample_Y(Y_val,i)))
    # clf.fit(np.log(X_train+1),sample_Y(Y_train,i))
    # print(clf.score(np.log(X_val+1),sample_Y(Y_val,i)))

    print('')




# clf.fit(np.log(X_train+1),Y_train)
# print(clf.score(np.log(X_val+1),Y_val))

# clf.fit(np.sqrt(X_train+1),Y_train)
# print(clf.score(np.sqrt(X_val+1),Y_val))

# sds = StandardScaler()

# import numpy as np


# A = np.arange(9).reshape(3,3)

# # B = np.arange(9).reshape(3,3)
# # print(Normalizer(norm='l2').fit_transform(A,axis=1)
# # )

# A_transform,norm = normalize(A,axis=0,norm='l2',return_norm=True)

# B = np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]]) 
# print(A_transform*norm)
# print(B/norm)

