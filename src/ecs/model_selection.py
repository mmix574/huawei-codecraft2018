import copy
import random

from linalg.common import mean, minus, shape, sqrt, square
from linalg.matrix import vstack
from metrics import official_score,l2_loss


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

def cross_val_score(estimator_instance,X,y,is_shuffle=False,cv='full',scoring='score',random_state=None,return_mean=False,verbose=False):
    assert((type(cv)==int and cv>1)or cv=='full')
    assert(scoring=='score' or scoring=='loss')
    
    if type(cv)==int:
        assert(cv<len(X))
    if is_shuffle:
        X,y = shuffle(X,y=y,random_state=random_state)
    N = len(X)
    K = N if cv=='full' else cv

    h = len(X)/float(K)

    scores = []
    losses = []
    for i in range(K):
        s = int(round((i*h)))
        e = int(round((i+1)*h))

        X_train,Y_train = [],[]
        X_train.extend(X[:s])
        X_train.extend(X[e:])
        Y_train.extend(y[:s])
        Y_train.extend(y[e:])

        X_val,Y_val = X[s:e],y[s:e]
        estimator_instance.fit(X_train,Y_train)
        
        p= estimator_instance.predict(X_val) 
        score = official_score(p,Y_val)
        loss = l2_loss(p,Y_val)
        # score = estimator_instance.score(X_val,Y_val)
        scores.append(score)
        losses.append(loss)

    # print(scores)
    if return_mean:
        if scoring=='score':
            std = sqrt(mean(square(minus(scores,mean(scores)))))
            # return (sorted(scores)[len(scores)/2] + mean(scores) - 0.5*std)/2.0
            # return (sorted(scores)[len(scores)/2] + mean(scores) - std)/2.0
            # return sorted(scores)[len(scores)/2] - std
            # return max(scores)
            return mean(scores)
            # return mean(sorted(scores)[::-1][:len(scores)/2])
            # return (mean(scores) + max(scores))/2.0
            # return mean(scores) - std
            # return mean(scores) -0.5*std
        elif scoring=='loss':
            # return mean(losses)
            std = sqrt(mean(square(minus(losses,mean(losses)))))
            # return mean(losses) + std
            return ((sorted(losses)[len(losses)/2] + mean(losses) + std)/2.0)

    else:
        if scoring=='score':
            return scores
        elif scoring=='loss':
            return losses


# support for score only
def grid_search_cv(estimator,paramaters,X,y,is_shuffle=False,cv='full',scoring='score',random_state=None,verbose=False,return_parameter=False):
    assert(scoring=='score' or scoring=='loss')
    def paramater_gen(paramaters):
        N = len(paramaters)
        from itertools import product
        value = list(product(*paramaters.values()))
        for v in value:
            yield dict(zip(paramaters.keys(),v))

    max_model = None
    max_parameter = None
    max_score = None
    min_loss = None
    for p in paramater_gen(paramaters):
        clf = estimator(**p)
        clf.fit(X,y)
        score = cross_val_score(clf,X,y,return_mean=True,is_shuffle=is_shuffle,cv=cv,scoring=scoring,random_state=random_state) 
        # clf.score(X,y)
        if verbose:
            # print(p,score)
            pass
        
        if scoring == "score":
            if max_parameter==None or max_score<score:
                max_parameter = p
                max_score = score
                max_model = clf
        if scoring== "loss":
            if max_parameter==None or min_loss>score:
                max_parameter = p
                min_loss = score
                max_model = clf
    if verbose:
        print("max_parameter",max_parameter)

    if return_parameter:
        return max_model,max_parameter
    else:
        return max_model

