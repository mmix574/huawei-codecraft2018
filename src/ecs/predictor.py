import math
import random
import re
from datetime import datetime
from datetime import timedelta
from backpack import backpack_random_k_times
from linalg.common import (dim, dot, mean, multiply, reshape, shape, sqrt,
                           square, zeros,plus,sum,minus)
from linalg.common import flatten
from linalg.matrix import hstack, matrix_matmul, matrix_transpose, shift,matrix_copy
from linalg.vector import count_nonezero,arange

from utils import parse_ecs_lines, parse_input_lines
from utils import corrcoef
from utils import l2_loss,official_score

def get_flavors_unique_mapping(flavors_unique):
    mapping_index = {}.fromkeys(flavors_unique)
    c = 0
    for f in flavors_unique:
        mapping_index[f] = c
        c+=1
    return mapping_index


# fix bug 2018-04-02
# modify @2018-03-28
# sample[len(sample)-1] is latest frequency slicing 
def resample(ecs_logs,flavors_unique,training_start_time,predict_start_time,frequency='7d',weekday_align=None,N=1,get_flatten=False,argumentation=False):
    assert(frequency[len(frequency)-1]=='d')
    assert((weekday_align==None and argumentation==False) or (weekday_align==None and argumentation==True) or (weekday_align!=None and argumentation==False))


    if weekday_align != None:
        predict_start_time = predict_start_time - timedelta(days=(weekday_align.weekday()-predict_start_time.weekday()+7)%7)

    span_origin = int(frequency[:-1])
    training_days = ((predict_start_time - training_start_time).days) +1 

    span = 7 if span_origin<=7 else 14
    # print('training_days',training_days)
    max_sample_length = training_days//span

    sample = zeros((max_sample_length,len(flavors_unique)))

    mapping_index = get_flavors_unique_mapping(flavors_unique)

    for flavor,ecs_time in ecs_logs:
        ith = (predict_start_time - ecs_time).days//span

        # modify resample to align weekdays -- @2018-03-23
        if((predict_start_time - ecs_time).days%span>=span_origin):
            continue

        if ith >= len(sample):
            # filter some ecs_logs not enough for this frequency slicing
            continue
        if flavor not in flavors_unique:
            # filter some flavor not in the given flavors_unique
            continue
        # add count 
        sample[ith][mapping_index[flavor]] += 1

    sample = sample[::-1]
    
    def XY_generate(sample,N=1,get_flatten=False,return_test=False):
        X_train = []
        Y_train = []
        for i in range(N,len(sample)):
            X = [sample[i-N+k] for k in range(N)]
            if get_flatten:
                X = flatten(X)
            y = sample[i]
            X_train.append(X)
            Y_train.append(y)        
        if return_test:
            X_test = [sample[len(sample)-N+k] for k in range(N)]
            if get_flatten:
                X_test = flatten(X_test)
            return X_train,Y_train,X_test
        else:
            return X_train,Y_train

    X_train,Y_train,X_test = XY_generate(sample,N=N,get_flatten=get_flatten,return_test=True)
    return X_train,Y_train,X_test 


def predict_flavors_unique(ecs_logs,flavors_unique,training_start_time,training_end_time,predict_start_time,predict_end_time):
    # modify @ 2018-03-15 
    predict = {}.fromkeys(flavors_unique)
    for f in flavors_unique:
        predict[f] = 0
    virtual_machine_sum = 0

    mapping_index = get_flavors_unique_mapping(flavors_unique)
    predict_days = (predict_end_time-predict_start_time).days

    X_train,Y_train,X_test  = resample(ecs_logs,flavors_unique,training_start_time,predict_start_time,frequency='{}d'.format(predict_days),N=5,get_flatten=False)
    sm = Smoothing(weight_decay=0.4)
    sm.fit(X_train,Y_train)
    # print(sm.loss(X_train,Y_train))
    # print(sm.score(X_train,Y_train))

    model = grid_search(Smoothing,{"weight_decay":arange(0.1,0.99,100)},[X_train],[Y_train],[1],verbose=False)
    # from load_data import load_data
    # ll = load_data(flavors_unique,frequency='7d',weekday_align=None)

    # result = sm.predict(X_test)[0]
    result = model.predict(X_test)[0]
    for f in flavors_unique:
        p = result[mapping_index[f]]
        predict[f] = int(round(p))
        virtual_machine_sum += int(round(p))
    return predict,virtual_machine_sum


# using grid search to tune hyper paramaters
# estimator: regressor class
# paramaters = {'w':[0.1,0.2]},paramaters to try
def grid_search(estimator,paramaters,Xs,Ys,weights_of_samples,verbose=False,scoring="official"):
    def paramater_gen(paramaters):
        N = len(paramaters)
        from itertools import product
        value = list(product(*paramaters.values()))
        for v in value:
            yield dict(zip(paramaters.keys(),v))

    max_parameter = None
    
    max_score = None
    min_loss = None

    for p in paramater_gen(paramaters):
        clf = estimator(**p)
        clf.fit(Xs,Ys)
        
        score = clf.weighted_score(Xs,Ys,[1])
        loss = clf.weighted_loss(Xs,Ys,[1])

        if verbose:
            print(p,score,loss)
            
        assert(scoring == "official" or scoring == "l2loss")
        if scoring == "official":
            if max_parameter==None or max_score<score:
                max_parameter = p
                max_score = score
                min_loss = loss
        elif scoring == "l2loss":
            if max_parameter==None or min_loss>loss:
                max_parameter = p
                max_score = score
                min_loss = loss
    return estimator(**max_parameter)


# fix @ 2018-03-28
# predict last one or zero
class BasePredictor:
    def __init__(self):
        pass

    def fit(self,X,y):
        pass

    def weighted_fit(self,Xs,Ys,weights):
        pass

    def predict(self,X):
        if dim(X) == 1:
            return [0 for _ in X]
        R = [[0 for _ in range(shape(X)[1])]]
        for i in range(shape(X)[0]-1):
            R.append(X[i])
        return R

    def loss(self,X,y):
        y_ = self.predict(X)
        return l2_loss(y,y_)
    def score(self,X,y):
        y_ = self.predict(X)
        return official_score(y,y_)

    def weighted_loss(self,Xs,Ys,weights):
        total_loss = 0
        for i in range(len(Xs)):
            total_loss+=self.loss(Xs[i],Ys[i])
        return total_loss/float(len(Xs))

    def weighted_score(self,Xs,Ys,weights):
        total_score = 0
        for i in range(len(Xs)):
            total_score+=self.score(Xs[i],Ys[i])
        return total_score/float(len(Xs))


# add @2018-03-28
class Smoothing(BasePredictor):
    def __init__(self,weight_decay=0.4):
        BasePredictor.__init__(self)
        self.weight_decay = weight_decay

    def fit(self,X,y):
        pass

    def weighted_fit(self,Xs,Ys,weights):
        print('hello world')
        pass

    def predict(self,X):
        # assert(dim(X)==3)
        if dim(X)==2:
            X = [X]
        X_transform = [self.linear_weighted_smoothing(x) for x in X]
        return X_transform

    def linear_weighted_smoothing(self,X):
        assert(dim(X)==2)
        N = len(X)
        sum_w = sum([math.pow(self.weight_decay,k) for k in range(len(X))])
        
        R = X[0]
        for i in range(1,N):
            R = plus(
                multiply(X[i],math.pow(self.weight_decay,i)/float(sum_w)),
                R)
        return R


# build output lines
def predict_vm(ecs_lines,input_lines):
    if input_lines is None or ecs_lines is None:
        return []

    machine_config,flavors_number,flavors,flavors_unique,optimized,predict_start_time,predict_end_time = parse_input_lines(input_lines)
    ecs_logs,training_start_time,training_end_time = parse_ecs_lines(ecs_lines)

    predict,virtual_machine_sum = predict_flavors_unique(ecs_logs,flavors_unique,training_start_time,training_end_time,predict_start_time,predict_end_time)

    result = []
    result.append('{}'.format(virtual_machine_sum))
    for i in flavors_unique:
        result.append('flavor{} {}'.format(i,predict[i]))

    result.append('') # output '\n'

    # backpack_list,entity_machine_sum = backpack(machine_config,flavors,flavors_unique,predict)
    backpack_list,entity_machine_sum = backpack_random_k_times(machine_config,flavors,flavors_unique,predict,k=100)
    result.append('{}'.format(entity_machine_sum))

    # print(backpack_list)

    def _convert_machine_string(em):
        s = ""
        for k,v in em.items():
            if v != 0:
                s += " flavor{} {}".format(k,v)
        return s
    c = 1
    for em in backpack_list:
        result.append(str(c)+_convert_machine_string(em))
        c += 1

    # print(result)
    return result


