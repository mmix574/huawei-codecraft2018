import random

from linalg.common import shape

# add @ 2018-04-08
def train_test_split(X,y,test_size=0.2,random_state=None,align=None):
    assert(shape(X)[0]==shape(y)[0])

    N = shape(X)[0]

    if test_size>=1:
        test_length = test_size
    else:
        test_length = round(N*test_size)
        if test_length==0:
            test_length = 1

    if random_state!=None:
        random.seed(random_state)

    taining_length = N - test_length

    assert(align==None or align=='left' or align=='right')
    if align=='right':
        return X[:taining_length],X[taining_length:],y[:taining_length],y[taining_length:]
    elif align=='left':
        X[:test_length],X[test_length:],y[:test_length],y[test_length:]

    test_set = set()
    while len(test_set)!=test_length:
        i = random.randrange(N)
        if i not in test_set:
            test_set.add(i)

    X_train,X_test,Y_train,Y_test = [],[],[],[]

    for i in range(N):
        if i not in test_set:
            X_train.append(X[i])
            Y_train.append(y[i])
        else:
            X_test.append(X[i])
            Y_test.append(y[i])

    return X_train,X_test,Y_train,Y_test



def cross_val_score(estimator_instance,X,y,shuffle=False,cv='full',scoring='score'):
    

    pass