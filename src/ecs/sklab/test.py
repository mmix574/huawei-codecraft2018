from load_data import load_xy


N = 6

X_train,Y_train,X_val,Y_val = load_xy(prefix='../data/',argumentation = True,N=N)


from sklearn.linear_model import LinearRegression
# clf = LinearRegression()
# clf.fit(X_train,Y_train)

# print(clf.score(X_train,Y_train))


# print(clf.score(X_val,Y_val))
# print(X_train.shape)
# print(X_val.shape)


# from sklearn.ensemble import BaggingRegressor
# from sklearn.tree import DecisionTreeRegressor
# clf = BaggingRegressor(base_estimator=DecisionTreeRegressor())

from sklearn.linear_model import Lasso
from sklearn.cross_validation import cross_val_score
clf = LinearRegression()

def sample_X(X,i):
    K = X.reshape(X.shape[0],-1,20)
    return K[:,:,i:i+1].reshape(K.shape[0],-1)
def sample_Y(Y,i):
    return Y[:,i]


for i in range(Y_train.shape[1]):
    # print(sample_X(X_train,i))
    # print(sample_Y(Y_train,i))
    scores = cross_val_score(clf,sample_X(X_train,i),sample_Y(Y_train,i),cv=10)
    print(scores)
    # clf.fit(sample_X(X_train,i),sample_Y(Y_train,i))
    # fs = clf.score(sample_X(X_train,i),sample_Y(Y_train,i))
    # vs = clf.score(sample_X(X_val,i),sample_Y(Y_val,i))
    # print(fs,vs)


# for i in range(Y_train.shape[1]):
    # clf.fit(X_train[:,i:i+1],Y_train[:,i])
    # clf.fit(X_val,Y_val[:,i])
    # print(clf.score(X_train[:,i:i+1],Y_train[:,i]),clf.score(X_val[:,i:i+1],Y_val[:,i]))
    # print(clf.score())





