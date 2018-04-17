import sys 
sys.path.append('..')

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np

boston = load_boston()
X_train,X_test,Y_train,Y_test = train_test_split(boston.data,boston.target)

from learn.lasso import Lasso

X_train = [[ float(k) for k in x ] for x in X_train]
X_test = [[ float(k) for k in x ] for x in X_test]

Y_train = [float(k) for k in Y_train]
Y_test = [float(k) for k in Y_test]
# print(X_train,X_test)
# print(Y_train,Y_test)

print('')
clf = Lasso(n_iter=500)
clf.fit(X_train,Y_train)
p = clf.predict(X_train)
print(np.mean((np.array(p).reshape(-1,1)-np.array(Y_train))**2))
p = clf.predict(X_test)
print(np.mean((np.array(p).reshape(-1,1)-np.array(Y_test))**2))

print('')
from sklearn.linear_model import Lasso
clf = Lasso()
clf.fit(X_train,Y_train)
p = clf.predict(X_train)
print(np.mean((np.array(p).reshape(-1,1)-np.array(Y_train))**2))
p = clf.predict(X_test)
print(np.mean((np.array(p).reshape(-1,1)-np.array(Y_test))**2))