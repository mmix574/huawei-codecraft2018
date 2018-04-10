import math
import random
import re
from datetime import datetime, timedelta

from backpack import backpack_random_k_times
from learn.lasso import Lasso
from learn.linear_model import LinearRegression, Ridge
from linalg.common import (dim, dot, fancy, flatten, mean, minus, multiply,
                           plus, reshape, shape, sqrt, square, sum, zeros)
from linalg.matrix import (hstack, matrix_copy, matrix_matmul,
                           matrix_transpose, shift, vstack)
from linalg.vector import arange, count_nonezero

from utils import (get_flavors_unique_mapping,parse_ecs_lines, parse_input_lines)

from metrics import l2_loss,official_score
from linalg.matrix import corrcoef

from model_selection import cross_val_score, grid_search_cv, train_test_split
from predictions.base import BasePredictor
from preprocessing import normalize,minmax_scaling,standard_scaling,maxabs_scaling

from linalg.matrix import stdev
from linalg.vector import argsort

# fix bug 2018-04-02
# modify @2018-03-28
# sample[len(sample)-1] is latest frequency slicing 
# def resample(ecs_logs,flavors_unique,training_start_time,predict_start_time,frequency='7d',weekday_align=None,N=1,get_flatten=True,argumentation=False,outlier_handling=False):
#     assert(frequency[len(frequency)-1]=='d')
#     assert((weekday_align==None and argumentation==False) or (weekday_align==None and argumentation==True) or (weekday_align!=None and argumentation==False))

#     if type(weekday_align) == int:
#         predict_start_time = predict_start_time - timedelta(days=(weekday_align-predict_start_time.weekday()+7)%7)

#     elif weekday_align != None:
#         predict_start_time = predict_start_time - timedelta(days=(weekday_align.weekday()-predict_start_time.weekday()+7)%7)

#     if argumentation == True:
#         X_train,Y_train,X_test = [],[],[]
#         for i in range(7):
#             x,y,z = resample(ecs_logs,flavors_unique,training_start_time,predict_start_time,frequency='7d',weekday_align=i,N=N,get_flatten=get_flatten,argumentation=False)
#             X_train.extend(x)
#             Y_train.extend(y)
#             X_test = z            
#         return X_train,Y_train,X_test


#     span_origin = int(frequency[:-1])
#     training_days = ((predict_start_time - training_start_time).days) +1 

#     span = 7 if span_origin<=7 else 14
#     # print('training_days',training_days)
#     max_sample_length = training_days//span

#     sample = zeros((max_sample_length,len(flavors_unique)))

#     mapping_index = get_flavors_unique_mapping(flavors_unique)

#     for flavor,ecs_time in ecs_logs:
#         ith = (predict_start_time - ecs_time).days//span

#         # modify resample to align weekdays -- @2018-03-23
#         if((predict_start_time - ecs_time).days%span>=span_origin):
#             continue

#         if ith >= len(sample):
#             # filter some ecs_logs not enough for this frequency slicing
#             continue
#         if flavor not in flavors_unique:
#             # filter some flavor not in the given flavors_unique
#             continue
#         # add count 
#         sample[ith][mapping_index[flavor]] += 1

#     # --------------------------
#     sample = sample[::-1]
#     # [       old data            ]
#     # [                           ]
#     # [                           ]
#     # [                           ]
#     # [       new_data            ]
#     # --------------------------

#     # handling outlier
#     if outlier_handling:
#         def processing_sample(sample):
#             from preprocessing import stdev
#             m = mean(sample,axis=1)
#             std = stdev(sample)
#             # sample_T = matrix_transpose(sample)
#             removes = []
#             for i in range(shape(sample)[0]):
#                 for j in range(shape(sample)[1]):
#                     if abs(sample[i][j]-m[j]) > 2*std[j]:
#                         removes.append(i)
#                         # sample[i][j] = m[j]
#                         sample[i][j] = (1/3.0)*sample[i][j] + (2/3.0)*m[j]
#                         # sample[i][j] = (4/5.0)*sample[i][j] + (1/5.0)*m[j]
#                         # sample[i][j] = (7/8.0)*sample[i][j] + (1/8.0)*m[j]
#                 # sample = [sample[i] for i in range(len(sample)) if i not in removes]
#                 return sample
#         sample = processing_sample(sample)

#     def _XY_generate(sample,N=1,get_flatten=False,return_test=False):
        
#         X_train = []
#         Y_train = []
#         for i in range(N,len(sample)):
#             X = [sample[i-N+k] for k in range(N)]
#             if get_flatten:
#                 X = flatten(X)
#             y = sample[i]
#             X_train.append(X)
#             Y_train.append(y)

#         X_test = [sample[len(sample)-N+k] for k in range(N)]

#         if return_test:
#             if get_flatten:
#                 X_test = [flatten(X_test)]
#             return X_train,Y_train,X_test
#         else:
#             return X_train,Y_train
#     # end function

#     X_train,Y_train,X_test = _XY_generate(sample,N=N,get_flatten=get_flatten,return_test=True)
#     return X_train,Y_train,X_test 


# add @2018-04-10
# refactoring, do one thing
def resampling(ecs_logs,flavors_unique,training_start_time,predict_start_time,frequency=7,strike=3,skip=0):
    predict_start_time = predict_start_time-timedelta(days=skip)
    days_total = (predict_start_time-training_start_time).days

    sample_length = (days_total-frequency)/strike
    mapping_index = get_flavors_unique_mapping(flavors_unique)

    sample = zeros((sample_length,len(flavors_unique)))
    
    for i in range(sample_length):
        for f,ecs_time in ecs_logs:
            # 0 - 6 for example
            if (predict_start_time-ecs_time).days >=(i)*strike and (predict_start_time-ecs_time).days<(i+1)*strike+frequency:
                sample[i][mapping_index[f]] += 1
    # ----------------------------#
    sample = sample[::-1]
    # [       old data            ]
    # [                           ]
    # [                           ]
    # [                           ]
    # [       new_data            ]
    # ----------------------------#

    assert(shape(sample)==(sample_length,len(flavors_unique)))
    return sample

# # freezed prepare
# def normaling(X_train,Y_train,X_test,normalize_method='standard_scaling',norm='l1'):
#     assert(normalize_method=='normalize' or normalize_method=='minmax_scaling' or normalize_method=='standard_scaling')
#     N = shape(X_train)[0]
#     X = vstack([X_train,X_test])

#     if normalize_method=='normalize':
#         # 55.08
#         X = normalize(X,norm=norm)
#     elif normalize_method=='minmax_scaling':
#         # 56.308
#         X = minmax_scaling(X)
#     elif normalize_method=='standard_scaling':
#         # 59.661
#         X = standard_scaling(X)
    
#     X_train = X[:N]
#     X_test = X[N:]
#     return X_train,Y_train,X_test


def simple(ecs_logs,flavors_config,flavors_unique,training_start_time,training_end_time,predict_start_time,predict_end_time):
    predict = {}.fromkeys(flavors_unique)
    for f in flavors_unique:
        predict[f] = 0
    virtual_machine_sum = 0

    mapping_index = get_flavors_unique_mapping(flavors_unique)
    predict_days = (predict_end_time-predict_start_time).days
    
    N = 1

    X_train,Y_train,X_test  = resample(ecs_logs,flavors_unique,training_start_time,predict_start_time,frequency='{}d'.format(predict_days),N=N,argumentation=False,outlier_handling=True)
    # X_train,Y_train,X_test = normaling(X_train,Y_train,X_test,normalize_method='normalize')


    # 56.214
    result = mean(Y_train,axis=1)
    
    # 74.713
    # result = Y_train[-1]
    result = [0 if r<0 else r for r in result]
    for f in flavors_unique:
        p = result[mapping_index[f]]
        predict[f] = int(round(p))
        virtual_machine_sum += int(round(p))
    return predict,virtual_machine_sum

def smoothing(ecs_logs,flavors_config,flavors_unique,training_start_time,training_end_time,predict_start_time,predict_end_time):
    predict = {}.fromkeys(flavors_unique)
    for f in flavors_unique:
        predict[f] = 0
    virtual_machine_sum = 0

    mapping_index = get_flavors_unique_mapping(flavors_unique)
    predict_days = (predict_end_time-predict_start_time).days

    N = 1
    X_train,Y_train,X_test  = resample(ecs_logs,flavors_unique,training_start_time,predict_start_time,frequency='{}d'.format(predict_days),N=N,argumentation=False,outlier_handling=True)
    weight_decay = 0.5

    X = Y_train[::-1][:5]

    norm = sum([math.pow(weight_decay,k) for k in range(shape(X)[0])])
    W = [math.pow(weight_decay,k)/norm for k in range(shape(X)[0])]
    W = matrix_transpose([W for _ in range(shape(X)[1])])

    result = sum(multiply(X,W),axis=0)
    result = [0 if r<0 else r for r in result]
    for f in flavors_unique:
        p = result[mapping_index[f]]
        predict[f] = int(round(p))
        virtual_machine_sum += int(round(p))
    return predict,virtual_machine_sum


# 65.092
def ridge_full(ecs_logs,flavors_config,flavors_unique,training_start_time,training_end_time,predict_start_time,predict_end_time):
    predict = {}.fromkeys(flavors_unique)
    for f in flavors_unique:
        predict[f] = 0
    virtual_machine_sum = 0

    mapping_index = get_flavors_unique_mapping(flavors_unique)
    predict_days = (predict_end_time-predict_start_time).days
    
    N = 1

    X_train,Y_train,X_test  = resample(ecs_logs,flavors_unique,training_start_time,predict_start_time,frequency='{}d'.format(predict_days),N=N,argumentation=True,get_flatten=True)
    X = vstack([X_train,X_test])

    X_square = square(X)
    _sum = sum(X,axis=1)
    X_rate = [multiply(X[i],1/float(_sum[i])) if _sum[i]!=0 else X[i]  for i in range(shape(X)[0])]
    X_cube = [[math.pow(k,3) for k in x] for x in X]

    X = hstack([X])

    X = standard_scaling(X)
    X_train = X[:shape(X_train)[0]]
    X_test = X[shape(X_train)[0]:]

    # X_val = X_train[-predict_days:]
    # X_train = X_train[:-predict_days]
    # Y_val = Y_train[-predict_days:]
    # Y_train = Y_train[:-predict_days]

    clf = grid_search_cv(Ridge,{'alpha':[0.03,0.04,0.05,0.06,0.07,0.08,0.09,1,2,4,8,16]},X_train,Y_train,verbose=False)
    clf.fit(X_train,Y_train)
    result = clf.predict(X_test)[0]

    result = [0 if r<0 else r for r in result]
    for f in flavors_unique:
        p = result[mapping_index[f]]
        predict[f] = int(round(p))
        virtual_machine_sum += int(round(p))
    return predict,virtual_machine_sum

# 0.505056449986 validation_set score
def ridge_single(ecs_logs,flavors_config,flavors_unique,training_start_time,training_end_time,predict_start_time,predict_end_time):
    predict = {}.fromkeys(flavors_unique)
    for f in flavors_unique:
        predict[f] = 0
    virtual_machine_sum = 0

    mapping_index = get_flavors_unique_mapping(flavors_unique)
    predict_days = (predict_end_time-predict_start_time).days
    
    X_train,Y_train,X_test  = resample(ecs_logs,flavors_unique,training_start_time,predict_start_time,frequency='{}d'.format(predict_days),N=2)
    X = vstack([X_train,X_test])


    # feature engineering ..
    
    X = standard_scaling(X)
    X_train = X[:shape(X_train)[0]]
    X_test = X[shape(X_train)[0]:]

    X_val = X_train[-predict_days:]
    X_train = X_train[:-predict_days]
    Y_val = Y_train[-predict_days:]
    Y_train = Y_train[:-predict_days]

    clfs = []
    for i in range(shape(Y_train)[1]):
        clf = Ridge(alpha=1)
        X_ = reshape(X_train,(shape(X_train)[0],-1,shape(Y_train)[1]))
        X_ = fancy(X_,-1,-1,i)
        y_ = fancy(Y_train,-1,i)
        clf.fit(X_,y_)
        clfs.append(clf)
    result = []
    for i in range(shape(Y_train)[1]):
        X_ = reshape(X_test,(shape(X_test)[0],-1,shape(Y_train)[1]))
        X_ = fancy(X_,-1,-1,i)
        result.append(clfs[i].predict(X_))

    result = matrix_transpose(result)[0]
    result = [0 if r<0 else r*(predict_days/float(predict_days)) for r in result]
    for f in flavors_unique:
        p = result[mapping_index[f]]
        predict[f] = int(round(p))
        virtual_machine_sum += int(round(p))
    return predict,virtual_machine_sum


def corrcoef_supoort_ridge(ecs_logs,flavors_config,flavors_unique,training_start_time,training_end_time,predict_start_time,predict_end_time):
    predict = {}.fromkeys(flavors_unique)
    for f in flavors_unique:
        predict[f] = 0
    virtual_machine_sum = 0
    mapping_index = get_flavors_unique_mapping(flavors_unique)
    predict_days = (predict_end_time-predict_start_time).days

    N = 1
    X_train_raw,Y_train_raw,X_test  = resample(ecs_logs,flavors_unique,training_start_time,predict_start_time,frequency='{}d'.format(predict_days),N=N,argumentation=True,outlier_handling=False)
    X_train,X_val,Y_train,Y_val = train_test_split(X_train_raw,Y_train_raw,test_size=predict_days-1,align='right')
    corrcoef_of_data = corrcoef(X_train)
    

    X_train,Y_train,X_test = normaling(X_train,Y_train,X_test,normalize_method='standard_scaling')


    # not safe currently
    k = 1 if len(flavors_unique)<3 else 10

    assert(shape(corrcoef_of_data)[0]>=k)
    clfs = []
    indexes = []
    for i in range(shape(Y_train)[1]):
        col = corrcoef_of_data[i]
        col_index_sorted = argsort(col)[::-1]
        index = col_index_sorted[:k]
        indexes.append(index)
        clf = Ridge(alpha=1,fit_intercept=False)

        # from sklearn.svm import SVR
        # clf = SVR()
        
        w = fancy(col,index)
        X_ = fancy(X_train,-1,index)
        X_ = multiply(X_,w)
        
        clf = grid_search_cv(Ridge,{'alpha':[1,2,4,8,16]},X_,fancy(Y_train,-1,i),cv='full',verbose=False)
        # alpha = random.choice([0.000001,1,10,0.01,0.001])
        # clf = Ridge(alpha=alpha,fit_intercept=True)

        clf.fit(X_,fancy(Y_train,-1,i))
        clfs.append(clf)

    val_predict = []
    for i in range(shape(Y_train)[1]):
        X_ = fancy(X_val,-1,indexes[i])
        p = clfs[i].predict(X_)
        val_predict.append(p)
    val_predict = matrix_transpose(val_predict)

    # 0.0958653470916
    print(official_score(val_predict,X_val))

    # retraining on full dateset
    for i in range(shape(Y_train)[1]):
        col = corrcoef_of_data[i]
        w = fancy(col,indexes[i])
        X_ = fancy(X_train_raw,-1,indexes[i])
        X_ = multiply(X_,w)
        clf.fit(X_,fancy(Y_train_raw,-1,i))

    result = []
    for i in range(shape(Y_train)[1]):
        col = corrcoef_of_data[i]
        w = fancy(col,indexes[i])
        X_ = fancy(X_test,-1,indexes[i])
        X_ = multiply(X_,w)
        p = clfs[i].predict(X_)
        result.append(p)
    result = matrix_transpose(result)[0]


    result = [0 if r<0 else r for r in result]
    for f in flavors_unique:
        p = result[mapping_index[f]]
        predict[f] = int(round(p))
        virtual_machine_sum += int(round(p))
    return predict,virtual_machine_sum



# add @ 2018-04-09 
# zero padding feature extraction
# 74.103 online score 2018-04-09 
# 78.184
def features_building(ecs_logs,flavors_config,flavors_unique,training_start_time,training_end_time,predict_start_time,predict_end_time):
    predict = {}.fromkeys(flavors_unique)
    for f in flavors_unique:
        predict[f] = 0
    virtual_machine_sum = 0
    mapping_index = get_flavors_unique_mapping(flavors_unique)
    predict_days = (predict_end_time-predict_start_time).days

    strike = 2
    X = resampling(ecs_logs,flavors_unique,training_start_time,predict_start_time,frequency=predict_days,strike=strike,skip=0)
    # X_skip = resampling(ecs_logs,flavors_unique,training_start_time,predict_start_time,frequency=predict_days,strike=strike,skip=1)

    def outlier_handling(X):
        std_ = stdev(X)
        mean_ = mean(X,axis=0)
         
        for i in range(shape(X)[0]):
            for j in range(shape(X)[1]):
               
                pass
    
    def get_corrcorf_path(X,return_ith=False,k=3):
        coef_X = []
        corrcoef_X = corrcoef(X)
        for i in range(shape(X)[1]):
            col = corrcoef_X[i]
            col_index_sorted = argsort(col)[::-1]
            index = col_index_sorted[:k]
            coef_X.append(fancy(X,-1,index))
        return coef_X

    coef_X = get_corrcorf_path(X,k=2)
    
    Y = X[1:]
    X_trainS,Y_trainS,X_test_S = [],[],[]

    for f in flavors_unique:
        history = fancy(X,-1,mapping_index[f])
        
        feature_grid = []
        # feature_grid:
        # (n_samples,n_features) 
        # [-----------|----------1]
        # [-----------|-------1--2]
        # [-----------|----1--2--3]  --<--cut some where,and fill the blank with some value.
        # [-----------|-..........]
        # [1--2--3....|..........n]
        # sparse  --------  dense
        
        # building feature grid
        for i in range(len(history)):
            fea = []
            m = mean(history[:i+1])
            # fill with mean
            # fea.extend([m for _ in range(len(history)-1-i)])
            fea.extend([0 for _ in range(len(history)-1-i)])
            fea.extend(history[:i+1])
            feature_grid.append(fea)


        feature_grid = fancy(feature_grid,-1,(-4,))
        
        # print(shape(feature_grid))
        # print(shape(X))
        # exit()

        feature_grid_square = square(feature_grid)
        
        # feature_grid = hstack([feature_grid,coef_X[mapping_index[f]]])
        feature_grid = hstack([feature_grid,coef_X[mapping_index[f]]])
        feature_grid = hstack([feature_grid,feature_grid_square])

        # ..filter the sparse feature by checking stdev..
        # std = stdev(feature_grid)
        # # print(std)
        # drop = [False if s<=0.5 else True for s in std]
        # if count_nonezero(drop)/len(drop) <0.5:
        #     feature_grid = fancy(feature_grid,-1,drop)

        # ... other preprocessing ..
        # print(shape(feature_grid[:-shape(X_test_raw)[0]]))
        # print(shape(feature_grid[-shape(X_test_raw)[0]:]))
        # print(fancy(y,dim_0,mapping_index[f]))


        # feature_grid = standard_scaling(feature_grid)
        # feature_grid = normalize(feature_grid,norm='l1')
        # feature_grid = normalize(feature_grid,norm='l2')
        # feature_grid = minmax_scaling(feature_grid)
        feature_grid = maxabs_scaling(feature_grid)
    

        X_trainS.append(feature_grid[:-1])
        X_test_S.append(feature_grid[-1:])
        Y_trainS.append(fancy(Y,-1,mapping_index[f]))

    return X_trainS,Y_trainS,X_test_S

# offline:
# 0.555144042315
# online:
# 56.898
def merge(ecs_logs,flavors_config,flavors_unique,training_start_time,training_end_time,predict_start_time,predict_end_time):
    predict = {}.fromkeys(flavors_unique)
    for f in flavors_unique:
        predict[f] = 0
    virtual_machine_sum = 0
    mapping_index = get_flavors_unique_mapping(flavors_unique)
    predict_days = (predict_end_time-predict_start_time).days

    # N = 1
    # X_train_raw,Y_train_raw,X_test  = resample(ecs_logs,flavors_unique,training_start_time,predict_start_time,frequency='{}d'.format(predict_days),N=N,argumentation=True,outlier_handling=True)
    # X_train,X_val,Y_train,Y_val = train_test_split(X_train_raw,Y_train_raw,test_size=predict_days-1,align='right')

    # result = new_feature(X_train,Y_train,X_test)
    # result = [0 if r<0 else r for r in result]
    # for f in flavors_unique:
    #     p = result[mapping_index[f]]
    #     predict[f] = int(round(p))
    #     virtual_machine_sum += int(round(p))
    # return predict,virtual_machine_sum

    # ------------------------------------#
    # predict_method = simple
    # predict_method = smoothing
    # predict_method = ridge_single
    # predict_method = ridge_full
    # return predict_method(ecs_logs,flavors_config,flavors_unique,training_start_time,training_end_time,predict_start_time,predict_end_time)
    
    # predict_method = corrcoef_supoort_ridge
    # predict_method = features_ridge

    X_trainS,Y_trainS,X_test_S = features_building(ecs_logs,flavors_config,flavors_unique,training_start_time,training_end_time,predict_start_time,predict_end_time)
    result = [] 
    for f in flavors_unique:
        X = X_trainS[mapping_index[f]]
        y = Y_trainS[mapping_index[f]]
        X_test = X_test_S[mapping_index[f]]
        clf = grid_search_cv(Ridge,{'alpha':[0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,1,2,4,8,16,32,64],'fit_intercept':[True]},X,y,verbose=True,is_shuffle=False,scoring='score')
        # clf = Ridge(alpha=1,fit_intercept=True)
        clf.fit(X,y)


        result.append(clf.predict(X_test))

    result = matrix_transpose(result)[0]
    result = [0 if r<0 else r for r in result]
    for f in flavors_unique:
        p = result[mapping_index[f]]
        predict[f] = int(round(p))
        virtual_machine_sum += int(round(p))
    return predict,virtual_machine_sum
    # return predict_method(ecs_logs,flavors_unique,training_start_time,training_end_time,predict_start_time,predict_end_time)


# build output lines
def predict_vm(ecs_lines,input_lines):
    if input_lines is None or ecs_lines is None:
        return []

    machine_config,flavors_number,flavors_config,flavors_unique,optimized,predict_start_time,predict_end_time = parse_input_lines(input_lines)
    ecs_logs,training_start_time,training_end_time = parse_ecs_lines(ecs_lines,flavors_unique)

    predict,virtual_machine_sum = merge(ecs_logs,flavors_config,flavors_unique,training_start_time,training_end_time,predict_start_time,predict_end_time)

    result = []
    result.append('{}'.format(virtual_machine_sum))
    for i in flavors_unique:
        result.append('flavor{} {}'.format(i,predict[i]))

    result.append('') # output '\n'

    # backpack_list,entity_machine_sum = backpack(machine_config,flavors,flavors_unique,predict)
    backpack_list,entity_machine_sum = backpack_random_k_times(machine_config,flavors_config,flavors_unique,predict,optimized,k=1000)
    
    # todo 
    # from backpack import maximize_score_backpack
    # backpack_list,entity_machine_sum = maximize_score_backpack(machine_config,flavors,flavors_unique,predict,optimized,k=1000)

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
