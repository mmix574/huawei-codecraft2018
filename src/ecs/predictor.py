import math
import random
import re
from datetime import datetime
from datetime import timedelta
from backpack import backpack_random_k_times
from linalg.common import (dim, dot, mean, multiply, reshape, shape, sqrt,
                           square, zeros,plus,sum,minus)
from linalg.common import flatten,fancy
from linalg.matrix import hstack,vstack,matrix_matmul, matrix_transpose, shift,matrix_copy
from linalg.vector import count_nonezero,arange

from utils import parse_ecs_lines, parse_input_lines
from utils import corrcoef
from utils import l2_loss,official_score
from utils import get_flavors_unique_mapping

from learn.linear_model import LinearRegression,Ridge
from learn.lasso import Lasso

from predictions.base import BasePredictor
from preprocessing import normalize


# fix bug 2018-04-02
# modify @2018-03-28
# sample[len(sample)-1] is latest frequency slicing 
def resample(ecs_logs,flavors_unique,training_start_time,predict_start_time,frequency='7d',weekday_align=None,N=1,get_flatten=True,argumentation=False,outlier_handling=False):
    assert(frequency[len(frequency)-1]=='d')
    assert((weekday_align==None and argumentation==False) or (weekday_align==None and argumentation==True) or (weekday_align!=None and argumentation==False))

    if type(weekday_align) == int:
        predict_start_time = predict_start_time - timedelta(days=(weekday_align-predict_start_time.weekday()+7)%7)

    elif weekday_align != None:
        predict_start_time = predict_start_time - timedelta(days=(weekday_align.weekday()-predict_start_time.weekday()+7)%7)

    if argumentation == True:
        X_train,Y_train,X_test = [],[],[]
        for i in range(7):
            x,y,z = resample(ecs_logs,flavors_unique,training_start_time,predict_start_time,frequency='7d',weekday_align=i,N=N,get_flatten=get_flatten,argumentation=False)
            X_train.extend(x)
            Y_train.extend(y)
            X_test = z            
        return X_train,Y_train,X_test


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

    # handling outlier
    if outlier_handling:
        def processing_sample(sample):
            from preprocessing import stdev
            m = mean(sample,axis=1)
            std = stdev(sample)
            # sample_T = matrix_transpose(sample)
            removes = []
            for i in range(shape(sample)[0]):
                for j in range(shape(sample)[1]):
                    if abs(sample[i][j]-m[j]) > 3*std[j]:
                        removes.append(i)
                        sample[i][j] = m[j]
                        # sample[i][j] = (1/3.0)*sample[i][j] + (2/3.0)*m[j]
                        # sample[i][j] = (4/5.0)*sample[i][j] + (1/5.0)*m[j]
                        # sample[i][j] = (7/8.0)*sample[i][j] + (1/8.0)*m[j]
            return sample
            
            if len(removes)/float(len(sample)) <0.5:
                sample = [sample[i] for i in range(len(sample)) if i not in removes]
                return sample
            else:
                return sample
        sample = processing_sample(sample)

    def _XY_generate(sample,N=1,get_flatten=False,return_test=False):
        
        X_train = []
        Y_train = []
        for i in range(N,len(sample)):
            X = [sample[i-N+k] for k in range(N)]
            if get_flatten:
                X = flatten(X)
            y = sample[i]
            X_train.append(X)
            Y_train.append(y)

        X_test = [sample[len(sample)-N+k] for k in range(N)]

        if return_test:
            if get_flatten:
                X_test = [flatten(X_test)]
            return X_train,Y_train,X_test
        else:
            return X_train,Y_train
    # end function

    X_train,Y_train,X_test = _XY_generate(sample,N=N,get_flatten=get_flatten,return_test=True)
    return X_train,Y_train,X_test 



def predict_flavors_unique_Smoothing_gridsearch(ecs_logs,flavors_unique,training_start_time,training_end_time,predict_start_time,predict_end_time):
    predict = {}.fromkeys(flavors_unique)
    for f in flavors_unique:
        predict[f] = 0
    virtual_machine_sum = 0

    mapping_index = get_flavors_unique_mapping(flavors_unique)
    predict_days = (predict_end_time-predict_start_time).days
    
    N = 3
    # with argumentation
    X_train,Y_train,X_test  = resample(ecs_logs,flavors_unique,training_start_time,predict_start_time,frequency='{}d'.format(predict_days),N=N,get_flatten=True,argumentation=True)
    
    # from load_data import load_data
    # X_train_old,Y_train_old = load_data(flavors_unique,frequency='{}d'.format(predict_days),weekday_align=None,N=N,get_flatten=True,argumentation=False,which=[0,1])
    # X_train.extend(X_train_old)
    # Y_train.extend(Y_train_old)
    
    model = grid_search(Smoothing,{"weight_decay":arange(0.1,0.7,200)},X_train,Y_train,verbose=False)
    # model = Smoothing(weight_decay=0.73)
    model.fit(X_train,Y_train)
    result = model.predict(X_test)[0]
    for f in flavors_unique:
        p = result[mapping_index[f]]
        predict[f] = int(round(p))
        virtual_machine_sum += int(round(p))
    return predict,virtual_machine_sum


def ridge_full(ecs_logs,flavors_unique,training_start_time,training_end_time,predict_start_time,predict_end_time):
    predict = {}.fromkeys(flavors_unique)
    for f in flavors_unique:
        predict[f] = 0
    virtual_machine_sum = 0

    mapping_index = get_flavors_unique_mapping(flavors_unique)
    predict_days = (predict_end_time-predict_start_time).days
    
    N = 3
    get_flatten = True
      # with argumentation
    X_train,Y_train,X_test  = resample(ecs_logs,flavors_unique,training_start_time,predict_start_time,frequency='{}d'.format(predict_days),N=N,get_flatten=get_flatten,argumentation=True)

    # X_train.extend(X_train)
    # Y_train.extend(Y_train)

    from load_data import load_data
    X_train_old,Y_train_old = load_data(flavors_unique,frequency='{}d'.format(predict_days),weekday_align=predict_end_time,N=N,get_flatten=get_flatten,argumentation=False,which=[0,2])

    # X_train.extend(X_train_old)
    # Y_train.extend(Y_train_old)

    # X_new = vstack([X_train,X_train_old])
    # Y_new = vstack([Y_train,Y_train_old])
    # _,norm_X = normalize(X_new,norm='l1',axis=1,return_norm=True)
    # norm_inv_X = [0 if x==0 else 1/float(x)for x in norm_X]
    # _,norm_Y = normalize(Y_new,norm='l1',axis=1,return_norm=True)
    # norm_inv_Y = [0 if x==0 else 1/float(x)for x in norm_Y]
    # X_train = multiply(X_train,norm_inv_X)
    # X_train_old = multiply(X_train_old,norm_inv_X)

    # Y_train = multiply(Y_train,norm_inv_Y)
    # Y_train_old = multiply(Y_train_old,norm_inv_Y)

    X_train,norm_X = normalize(X_train,norm='l1',axis=1,return_norm=True)
    norm_inv_X = [0 if x==0 else 1/float(x)for x in norm_X]
    X_test = multiply(X_test,norm_inv_X)

    ridge = Ridge_Full(alpha=1)
    ridge.fit(X_train,Y_train)
    result = ridge.predict(X_test)[0]


    result = [0 if r<0 else r for r in result]
    for f in flavors_unique:
        p = result[mapping_index[f]]
        predict[f] = int(round(p))
        virtual_machine_sum += int(round(p))
    return predict,virtual_machine_sum


# class Multi_Prediction_Warpper:
#     def __init__(self,estimator,parameter):
#         self.estimator = estimator
#         self.parameter = parameter
#         self.clfs = []
#         self.shape_Y = None

#     def fit(self,X,y):
#         assert(dim(y)==2)
        
#         self.shape_Y = shape(y)
#         for i in range(shape(y)[1]):
#             p = self.parameter
#             clf = self.estimator(**p)
#             clf.fit(X,fancy(y,-1,i))
#             self.clfs.append(clf)

#     def predict(self,X):
#         R = []
#         for i in range(self.shape_Y):
#             predict = self.clfs[i].predict(X)
#             R.append(predict)
#         return matrix_transpose(R)



def ridge_single(ecs_logs,flavors_unique,training_start_time,training_end_time,predict_start_time,predict_end_time):
    predict = {}.fromkeys(flavors_unique)
    for f in flavors_unique:
        predict[f] = 0
    virtual_machine_sum = 0

    mapping_index = get_flavors_unique_mapping(flavors_unique)
    predict_days = (predict_end_time-predict_start_time).days
    
    N = 4

    X_train,Y_train,X_test  = resample(ecs_logs,flavors_unique,training_start_time,predict_start_time,frequency='{}d'.format(predict_days),N=N,argumentation=False,get_flatten=True)

    from load_data import load_data
    X_train_old,Y_train_old = load_data(flavors_unique,frequency='{}d'.format(predict_days),weekday_align=predict_end_time,N=N,argumentation=False,which=[0,2],get_flatten=True)
    

    # from preprocessing import normalize
    # X_train,norm = normalize(X_train,norm='l1',axis=1,return_norm=True)
    # norm_inv = [0 if x==0 else 1/float(x)for x in norm]
    # X_test = multiply(X_test,norm_inv)

    ridge = Ridge_Single(alpha=1)
    ees = bagging_estimator(Ridge_Single,{'alpha':1},max_clf=1000)

    # X,y = bagging_with_model(ees,X_train_old,Y_train_old,X_train,Y_train,max_iter=1000,scoring='score')
    # X.extend(X_train)
    # y.extend(Y_train)

    ees.fit(X_train,Y_train)
    # ridge.fit(X_train,Y_train)
    
    result = ees.predict(X_test)[0]
    # result = ridge.predict(X_test)[0]

    result = [0 if r<0 else r for r in result]
    # result = model.predict(X_test)[0]
    for f in flavors_unique:
        p = result[mapping_index[f]]
        predict[f] = int(round(p))
        virtual_machine_sum += int(round(p))
    return predict,virtual_machine_sum





def corrcoef_supoort_ridge_single(ecs_logs,flavors_unique,training_start_time,training_end_time,predict_start_time,predict_end_time):
    predict = {}.fromkeys(flavors_unique)
    for f in flavors_unique:
        predict[f] = 0
    virtual_machine_sum = 0
    mapping_index = get_flavors_unique_mapping(flavors_unique)
    predict_days = (predict_end_time-predict_start_time).days

    N = 1
    X_train,Y_train,X_test  = resample(ecs_logs,flavors_unique,training_start_time,predict_start_time,frequency='{}d'.format(predict_days),N=N,argumentation=True)
    from load_data import load_data
    X_train_old,Y_train_old = load_data(flavors_unique,frequency='{}d'.format(predict_days),weekday_align=predict_end_time,N=N,argumentation=False,which=[0,2])
    
    from preprocessing import normalize

    X_train,norm_inv = normalize(X_train,norm='l1',axis=1,return_norm_inv=True)
    X_test = multiply(X_test,norm_inv)

    
    ridge = Ridge_Single(alpha=1)

    # from utils import corrcoef
    # X_train_old,Y_train_old = load_data(flavors_unique,frequency='{}d'.format(1),weekday_align=None,N=N,argumentation=False,which=[0,1,2])
    # corrcoef_ = corrcoef(X_train)
    # from preprocessing import normalize
    # corrcoef_ = normalize(corrcoef_,norm='l1')
    # X_train_new = matrix_matmul(X_train,corrcoef_)
    # X_train = hstack([X_train_new,X_train])
    # ridge = Ridge_Single(alpha=1)
    # ridge.fit(X_train,Y_train)
    # X_test_new = matrix_matmul(X_test,corrcoef_)
    # X_test = hstack([X_test,X_test_new])

    ridge.fit(X_train,Y_train)

    result = ridge.predict(X_test)[0]
    result = [0 if r<0 else r for r in result]
    for f in flavors_unique:
        p = result[mapping_index[f]]
        predict[f] = int(round(p))
        virtual_machine_sum += int(round(p))
    return predict,virtual_machine_sum


def random_feature_selection_ridge_full(ecs_logs,flavors_unique,training_start_time,training_end_time,predict_start_time,predict_end_time):
    predict = {}.fromkeys(flavors_unique)
    for f in flavors_unique:
        predict[f] = 0
    virtual_machine_sum = 0
    mapping_index = get_flavors_unique_mapping(flavors_unique)
    predict_days = (predict_end_time-predict_start_time).days

    N = 2
    X_train_raw,Y_train_raw,X_test_raw  = resample(ecs_logs,flavors_unique,training_start_time,predict_start_time,frequency='{}d'.format(predict_days),N=N,argumentation=True)
    
    from preprocessing import normalize
    X_train_raw,norm_inv = normalize(X_train_raw,norm='l1',axis=1,return_norm_inv=True)
    X_test_raw = multiply(X_test_raw,norm_inv)


    from model_selection import train_test_split

    X_train,X_Val,Y_train,Y_val = train_test_split(X_train_raw,Y_train_raw,test_size=predict_days-1,align='right')

    clf = Ridge_Full(alpha=0.01)
    clf.fit(X_train,Y_train)

    # from model_selection import cross_val_score,grid_search_cv
    # clf = grid_search_cv(Ridge_Single,{'alpha':[1,2,3,4,5]},X_train_raw,Y_train_raw,verbose=True)

    # dropout_estimator
    clf = random_feature_estimator(Ridge_Full,{'alpha':1},drop_out=0.7,max_iter=200)
    clf.train(X_train,Y_train,X_Val,Y_val)
    clf.retrain(X_train_raw,Y_train_raw)


    result = clf.predict(X_test_raw)[0]
    result = [0 if r<0 else r for r in result]
    for f in flavors_unique:
        p = result[mapping_index[f]]
        predict[f] = int(round(p))
        virtual_machine_sum += int(round(p))
    return predict,virtual_machine_sum




class random_feature_estimator(BasePredictor):
    def __init__(self,estimator,parameter,max_iter=20,drop_out=0.3,scoring='score'):
        self.estimator = estimator
        self.parameter = parameter
        self.max_iter = max_iter
        self.drop_out = drop_out
        self.scoring = scoring

        self.keeps = []
        self.clfs = []
        self.shape_Y = None

    def _rand_X(self,X):
        N = shape(X)[1]
        keep_length = math.ceil((1-self.drop_out)*N)
        keep_set = set()
        while len(keep_set)!=keep_length:
            i = random.randrange(N)
            if i not in keep_set:
                keep_set.add(i)
        keep = [True if i in keep_set else False for i in range(N)]
        X_ = fancy(X,-1,keep)
        return X_,keep

    def _get_keep_X(self,X,keep):
        return fancy(X,-1,keep)


    def train_cv(self,X,y,shuffle=False,cv='full'):
        assert(type(cv)==int or cv=='full')
        assert(dim(X)==2 and dim(y)==2)
        self.shape_Y = shape(y)



        # pass

    def train(self,X,y,X_val,Y_val):
        assert(dim(X)==2 and dim(y)==2)
        self.shape_Y = shape(y)

        for i in range(shape(y)[1]):
            max_score = None
            best_clf = None
            best_keep = None
            y_ = fancy(y,-1,i)
            for _ in range(self.max_iter):
                clf = self.estimator(**(self.parameter))
                X_,keep = self._rand_X(X)
                clf.fit(X_,y_)
                score = clf.score(self._get_keep_X(X_val,keep),fancy(Y_val,-1,i))

                if not max_score or max_score<score:
                    max_score = score
                    best_clf = clf
                    best_keep = keep

            self.keeps.append(best_keep)
            self.clfs.append(best_clf) 

    def retrain(self,X,y):
        assert(len(self.keeps)!=0)
        for i in range(self.shape_Y[1]):
            X_ = self._get_keep_X(X,self.keeps[i])
            self.clfs[i].fit(X_,fancy(y,-1,i))

    def predict(self,X):
        assert(dim(X)==2)
        result = []
        for i in range(self.shape_Y[1]):
            X_ = self._get_keep_X(X,self.keeps[i])
            result.append(self.clfs[i].predict(X_))
        return matrix_transpose(result)




class dropout_estimator(BasePredictor):
    def __init__(self,estimator,parameter,drop_out=0.7):
        self.estimator = estimator
        self.parameter = parameter
        self.drop_out = drop_out

        self.shape_X = None
        self.shape_Y = None
        self.clf = None
        self.keep = None

    def fit(self,X,y,keep_hyper=False):
        assert(dim(y)==1)
        self.shape_X = shape(X)
        self.shape_Y = shape(y)

        if not keep_hyper:
            keep = [True if random.random()>self.drop_out else False for _ in range(shape(X)[1])]
            self.keep = keep
            X_ = fancy(X,-1,keep)
            clf = self.estimator(**(self.parameter))
            clf.fit(X_,y)
            self.clf = clf
        else:
            keep = self.keep
            X_ = fancy(X,-1,keep)
            clf = self.estimator(**(self.parameter))
            clf.fit(X_,y)
            self.clf = clf

    def predict(self,X):
        keep = self.keep
        X_ = fancy(X,-1,keep)
        clf = self.clf
        return clf.predict(X_)



class bagging_estimator(BasePredictor):
    def __init__(self,estimator,parameter,max_clf = 100):
        self.estimator = estimator
        self.parameter = parameter
        self.max_clf = max_clf
        self.clfs = []

    def bagging(self,X_train,Y_train):
        N = shape(X_train)[0]
        index = []
        for i in range(N):
            index.append(random.randrange(N))
        X = [X_train[i] for i in index]
        y = [Y_train[i] for i in index]
        return X,y
    
    def fit(self,X,y):
        for i in range(self.max_clf):
            clf = self.estimator(**self.parameter)
            _X,_y = self.bagging(X,y)
            clf.fit(_X,y)
            self.clfs.append(clf)
        
    def predict(self,X):
        prediction = None
        for i in range(self.max_clf):
            if not prediction:
                prediction = self.clfs[i].predict(X)
            else:
                prediction = plus(prediction,self.clfs[i].predict(X))
        prediction = multiply(prediction,1/float(self.max_clf))
        return prediction



# add @2018-03-28
class Smoothing(BasePredictor):
    def __init__(self,weight_decay=0.4):
        BasePredictor.__init__(self)
        self.weight_decay = weight_decay
        self.shape_X = None
        self.shape_Y = None

    def fit(self,X,y):
        self.shape_X = shape(X)
        self.shape_Y = shape(y)

    def predict(self,X):
        X = reshape(X,(shape(X)[0],-1,self.shape_Y[-1]))
        X = X[::-1]
        N = shape(X)[1]

        norm = sum([math.pow(self.weight_decay,k) for k in range(N)])
        W = [math.pow(self.weight_decay,k)/norm for k in range(N)]
        W = [W for _ in range(self.shape_Y[-1])]
        W = matrix_transpose(W)
        R = []
        for i in range(shape(X)[0]):
            R.append(sum(multiply(X[i],W),axis=0))
        return R

class Ridge_Full(BasePredictor):
    def __init__(self,alpha=1):
        BasePredictor.__init__(self)
        self.clf = None
        self.alpha = alpha
    
    def fit(self,X,y):
        
        clf = Ridge(fit_intercept=True,alpha=self.alpha)
        X = reshape(X,(shape(X)[0],-1))
        clf.fit(X,y)
        self.clf = clf

    def predict(self,X):
        X = reshape(X,(shape(X)[0],-1))
        return self.clf.predict(X)


class Ridge_Single(BasePredictor):
    def __init__(self,alpha=1):
        BasePredictor.__init__(self)
        self.clfs = []
        self.alpha = alpha
        self.shape_X = None
    
    def fit(self,X,y):
        from linalg.common import fancy
        X = reshape(X,(shape(X)[0],-1,shape(y)[1]))
        self.shape_X = shape(X)
        for i in range(shape(y)[1]):
            # print(i)

            clf = Ridge(fit_intercept=True,alpha=self.alpha)
            _X = fancy(X,-1,-1,i)
            _y = fancy(y,-1,(i,i+1))
            
            clf.fit(_X,_y)
            self.clfs.append(clf)

    def predict(self,X):
        prediction = []
        from linalg.common import fancy
        s = list(self.shape_X)
        s[0] = -1
        X = reshape(X,s)
        for i in range(self.shape_X[-1]):
            clf = self.clfs[i]
            _X = fancy(X,-1,-1,i)
            p = clf.predict(_X)
            prediction.append(p)
        R = reshape(prediction,(shape(prediction)[0],-1))
        return matrix_transpose(R)
    


class Dynamic_Ridge_Single(BasePredictor):
    def __init__(self,alpha=1):
        BasePredictor.__init__(self)
        self.clfs = []
        self.alpha = alpha
        self.shape_X = None
    
    def fit(self,X,y):
        from linalg.common import fancy
        X = reshape(X,(shape(X)[0],-1,shape(y)[1]))
        self.shape_X = shape(X)
        for i in range(shape(y)[1]):
            # print(i)
            
            clf = Ridge(fit_intercept=True,alpha=self.alpha)
            _X = fancy(X,-1,-1,i)
            _y = fancy(y,-1,(i,i+1))
            
            clf.fit(_X,_y)
            self.clfs.append(clf)

    def predict(self,X):
        prediction = []
        from linalg.common import fancy
        s = list(self.shape_X)
        s[0] = -1
        X = reshape(X,s)
        for i in range(self.shape_X[-1]):
            clf = self.clfs[i]
            _X = fancy(X,-1,-1,i)
            p = clf.predict(_X)
            prediction.append(p)
        R = reshape(prediction,(shape(prediction)[0],-1))
        return matrix_transpose(R)




# build output lines
def predict_vm(ecs_lines,input_lines):

    # predict_method = predict_flavors_unique_Smoothing_gridsearch
    # predict_method = ridge_full
    # predict_method = ridge_single

    # predict_method = corrcoef_supoort_ridge_single

    predict_method = random_feature_selection_ridge_full



    if input_lines is None or ecs_lines is None:
        return []

    machine_config,flavors_number,flavors,flavors_unique,optimized,predict_start_time,predict_end_time = parse_input_lines(input_lines)
    ecs_logs,training_start_time,training_end_time = parse_ecs_lines(ecs_lines)

    predict,virtual_machine_sum = predict_method(ecs_logs,flavors_unique,training_start_time,training_end_time,predict_start_time,predict_end_time)

    result = []
    result.append('{}'.format(virtual_machine_sum))
    for i in flavors_unique:
        result.append('flavor{} {}'.format(i,predict[i]))

    result.append('') # output '\n'

    # backpack_list,entity_machine_sum = backpack(machine_config,flavors,flavors_unique,predict)
    backpack_list,entity_machine_sum = backpack_random_k_times(machine_config,flavors,flavors_unique,predict,k=1000)
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


