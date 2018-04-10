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
    Y = X[1:]

    # 75.581
    def outlier_handling(X):
        std_ = stdev(X)
        mean_ = mean(X,axis=0)
        for i in range(shape(X)[0]):
            for j in range(shape(X)[1]):
               if X[i][j]-mean_[j] >3*std_[j]:
                   X[i][j] = mean_[j]
                #    X[i][j] = sqrt(X[i][j])
        return X
    X = outlier_handling(X)

    def get_corrcorf_path(X,return_ith=False,k=3,return_path=True):
        coef_X = []
        paths = []
        corrcoef_X = corrcoef(X)
        for i in range(shape(X)[1]):
            col = corrcoef_X[i]
            col_index_sorted = argsort(col)[::-1]
            index = col_index_sorted[1:k]
            coef_X.append(fancy(X,-1,index))
            paths.append(index)

        if return_path:
            return coef_X,paths
        else:
            return coef_X

    coef_X,paths = get_corrcorf_path(X,return_path=True)

    def get_rate_X(X):
        sum_ = sum(X,axis=1)
        return [X[i] if sum_[i]==0 else multiply(X[i],1/float(sum_[i])) for i in range(shape(X)[0])]

    rate_X = get_rate_X(X)

    # def get_rate_X(X,i):
    #     pass

    # def get_cpu_rate(X):
    #     pass
    
    # def get_men_rate(X):
    #     pass
    diff_X = minus(X,shift(X,1,fill=0))
    diff_X_2 = minus(X,shift(X,2,fill=0))
    
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

        feature_grid = fancy(feature_grid,-1,(shape(feature_grid)[0]/4,shape(feature_grid)[0]))
        feature_grid_normalized = normalize(feature_grid,norm='l1')

        feature_grid_square = square(feature_grid)

        from linalg.common import apply
        feature_grid_log1p = apply(feature_grid,lambda x:math.log1p(x))
        # feature_grid_3 = apply(feature_grid_normalized,lambda x:math.pow(x,2))
        feature_grid_sqrt = sqrt(feature_grid)
        
        feature_grid = hstack([feature_grid,feature_grid_sqrt,feature_grid_log1p,feature_grid_square])
        feature_grid = hstack([feature_grid,  coef_X[mapping_index[f]]])
        # feature_grid = hstack([feature_grid,  coef_X[mapping_index[f]] , fancy(rate_X,-1,(mapping_index[f],mapping_index[f]+1))])
        
        def get_col(X,index):
            if type(index)==list:
                return fancy(X,-1,index)
            else:
               return fancy(X,-1,(index,index+1))
        
        feature_grid = hstack([feature_grid,get_col(diff_X,mapping_index[f])])
        feature_grid = hstack([feature_grid,get_col(diff_X_2,mapping_index[f])])
        
        # head = 10
        # ..filter the sparse feature by checking stdev..
        # std = stdev(feature_grid)
        # drop = [False if s<=0.25 else True for s in std]
        # if count_nonezero(drop)/len(drop) <0.5:
        #     feature_grid = fancy(feature_grid,-1,drop)

        # ... other preprocessing ..
        # print(shape(feature_grid[:-shape(X_test_raw)[0]]))
        # print(shape(feature_grid[-shape(X_test_raw)[0]:]))
        # print(fancy(y,dim_0,mapping_index[f]))


        feature_grid = normalize(feature_grid,norm='l1') #0.572884203014 0.593898128226  0.716294303441
        # feature_grid = normalize(feature_grid,norm='l2') #0.564134720247 0.620964076788 0.664466379897

        # feature_grid = minmax_scaling(feature_grid) #0.498931703737 0.519933525422 (test good) 0.794159038136
        
        # feature_grid = maxabs_scaling(feature_grid) #0.494858535164 0.589982815641 0.787027762262
        # feature_grid = standard_scaling(feature_grid) #0.492486341694 0.441285064425 0.579265952486
    

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
        # clf = grid_search_cv(Ridge,{'alpha':[0.01,0.001,0.0001,1e-5,1,2,4,8,16,32],'fit_intercept':[True]},X,y,verbose=True,is_shuffle=False,scoring='loss')
        clf = Ridge(alpha=1,fit_intercept=False)
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
