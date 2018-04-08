import random
import copy
from linalg.common import shape

def shuffle(X,y=None,random_state=None):
    if random_state != None:
        random.seed(random_state)
    new_X = []
    new_Y = []
    index = list(range(len(X)))
    while(len(index)!=0):
        i = random.choice(index)
        new_X.append(X[i])
        if y != None:
            new_Y.append(y[i])
        index.remove(i)

    if y!=None:
        return new_X,new_Y
    else:
        return new_X

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

# todo
def cross_val_score(estimator_instance,X,y,shuffle=False,cv='full',scoring='score',random_state=None):
    assert((type(cv)==int and cv>1)or cv=='full')
    if type(cv)==int:
        
        pass
    if shuffle:
        X,y = shuffle(X,y=y,random_state=random_state)
    N = len(X)
    K = N if cv=='full' else cv



