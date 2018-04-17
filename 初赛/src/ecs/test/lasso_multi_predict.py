import sys 
sys.path.append('..')

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np

boston = load_boston()
X_train,X_test,Y_train,Y_test =train_test_split(boston.data,boston.target)

X_train = X_train.tolist()
Y_train = Y_train.tolist()

from learn.lasso import Lasso

clf = Lasso()
clf.fit(X_train,Y_train)

p = clf.predict(X_train)
