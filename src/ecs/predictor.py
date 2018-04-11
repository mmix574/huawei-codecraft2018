import math
import random
import re
from datetime import datetime, timedelta

from backpack import backpack_random_k_times
from learn.knn import KNN_Regressor,Dynamic_KNN_Regressor
from learn.lasso import Lasso
from learn.ridge import Ridge
from learn.linear_model import LinearRegression
from linalg.common import (apply, dim, dot, fancy, flatten, mean, minus,
                           multiply, plus, reshape, shape, sqrt, square, sum,
                           zeros)
from linalg.matrix import (corrcoef, hstack, matrix_copy, matrix_matmul,
                           matrix_transpose, shift, stdev, vstack)
from linalg.vector import arange, argsort, count_nonezero
from metrics import l2_loss, official_score
from model_selection import cross_val_score, grid_search_cv, train_test_split
from predictions.base import BasePredictor
from preprocessing import (maxabs_scaling, minmax_scaling, normalize,
                           standard_scaling)
from utils import (get_flavors_unique_mapping, parse_ecs_lines,
                   parse_input_lines)

from ensemble import bagging_estimator

# add @2018-04-10
# refactoring, do one thing.
def resampling(ecs_logs,flavors_unique,training_start_time,predict_start_time,frequency=7,strike=3,skip=0):
    predict_start_time = predict_start_time-timedelta(days=skip)
    days_total = (predict_start_time-training_start_time).days

    sample_length = (days_total-frequency)/strike
    mapping_index = get_flavors_unique_mapping(flavors_unique)

    sample = zeros((sample_length,len(flavors_unique)))
    
    for i in range(sample_length):
        for f,ecs_time in ecs_logs:
            # 0 - 6 for example
            # fix serious bug @ 2018-04-11
            if (predict_start_time-ecs_time).days >=(i)*strike and (predict_start_time-ecs_time).days<(i)*strike+frequency:
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
# X:
#   f1 f2 f3 f4 f5 f6 ...
# t1[---------------------]
# t2[---------------------]
# t3[---------------------]
# t4[---------------------]
# ..[---------------------]

# feature_grid: 
# (n_samples,n_features) 
# [-----------|----------1]
# [-----------|-------1--2]
# [-----------|----1--2--3]  --<--cut some where,or fill the Na with some value.
# [-----------|-..........]
# [1--2--3....|..........n]
# sparse feature--  dense feature


def features_building(ecs_logs,flavors_config,flavors_unique,training_start_time,training_end_time,predict_start_time,predict_end_time,outlier_handeling=True):
    mapping_index = get_flavors_unique_mapping(flavors_unique)
    predict_days = (predict_end_time-predict_start_time).days

    strike = 1
    X = resampling(ecs_logs,flavors_unique,training_start_time,predict_start_time,frequency=predict_days,strike=strike,skip=0)
    Y = X[1:]

    def outlier_handling(sample,method='mean'):
        assert(method=='mean' or method=='zero')
        X = matrix_copy(sample)
        std_ = stdev(sample)
        mean_ = mean(sample,axis=1)
        for i in range(shape(X)[0]):
            for j in range(shape(X)[1]):
               if X[i][j]-mean_[j] >3*std_[j]:
                    if method=='mean':
                        X[i][j] = mean_[j]
                    elif method=='zero':
                        X[i][j] = 0
        return X
    
    if outlier_handeling:
        X = outlier_handling(X,method='mean')
    
    def flavor_clustering(X,k=3,variance_threshold=None):
        corrcoef_X = corrcoef(X)
        clustering_paths = []
        coef_X = []
        for i in range(shape(X)[1]):
            col = corrcoef_X[i]
            col_index_sorted = argsort(col)[::-1]
            if variance_threshold!=None:
                index = [i  for i in col_index_sorted if col[i]>variance_threshold]
            else:
                index = col_index_sorted[1:k+1]
            coef = fancy(col,index)
            coef_X.append(coef)

            clustering_paths.append(index)
        return clustering_paths,coef_X

    clustering_paths,coef_X = flavor_clustering(X)


    def get_feature_grid(X,i,fill_na='mean',max_na_rate=1):
        assert(fill_na=='mean' or fill_na=='zero')
        

        pass

    # 2018-04-12
    # handcraft feature engineering ..       
    # -------------------
    from utils import get_machine_config
    def get_rate_X(X):
        pass
    #     sum_ = sum(X,axis=1)
    #     return [X[i] if sum_[i]==0 else multiply(X[i],1/float(sum_[i])) for i in range(shape(X)[0])]
    # rate_X = get_rate_X(X)


    # def get_cpu_X(X):
    #     cpu_config,mem_config = get_machine_config(flavors_unique)

    #     return X


    X_trainS,Y_trainS,X_test_S = [],[],[]

    for f in flavors_unique:
        history = fancy(X,None,mapping_index[f])
        y = fancy(Y,None,mapping_index[f])
        feature_grid = []

        
        # building feature grid
        for i in range(len(history)):
            fea = []
            # if len(history[:i+1])==0:
            #     m = 0
            # else:
            #     m = sorted(history[:i+1])[len(history[:i+1])/2]

            m = mean(history[:i+1])
            fea.extend([m for _ in range(len(history)-1-i)])
            # fea.extend([0 for _ in range(len(history)-1-i)])
            fea.extend(history[:i+1])
            feature_grid.append(fea)

        feature_grid = fancy(feature_grid,None,(-4,))

        # max_zero_percent = 1
        # keep = []
        # feature_grid_T = matrix_transpose(feature_grid)
        # for col in feature_grid_T:
        #     if (1-(count_nonezero(col))/float(len(col)))<max_zero_percent :
        #         keep.append(True)
        #     else:
        #         keep.append(False)
        # feature_grid = fancy(feature_grid,-1,keep)

        feature_grid_log1p = apply(feature_grid,lambda x:math.log1p(x))
        feature_grid_sqrt = sqrt(feature_grid)
        feature_grid_square = square(feature_grid)
        add_list= [feature_grid] #	77.804

        add_list.extend([feature_grid_sqrt])
        add_list.extend([feature_grid_log1p]) # 77.998
        add_list.extend([feature_grid_square])
        # add_list.extend([coef_X[mapping_index[f]]])
        # add_list.extend([fancy(rate_X,None,(mapping_index[f],mapping_index[f]+1))])
        feature_grid = hstack(add_list)

        # exit()
        # ---------------------------------------------
        # ..filter the sparse feature by checking stdev..
        # std = stdev(feature_grid)
        # m = sorted(std)[int(len(std)*(1/float(2.0)))]
        # keep = [False if s<m else True for s in std]
        # feature_grid = fancy(feature_grid,-1,keep)



        # ---------------------------------------------
        feature_grid = normalize(feature_grid,norm='l1')
        # feature_grid = normalize(feature_grid,norm='l2')
        # feature_grid = minmax_scaling(feature_grid) 
        # feature_grid = maxabs_scaling(feature_grid)
        # feature_grid = standard_scaling(feature_grid)

        # rescaling_list = [normalize(feature_grid,norm='l1'),normalize(feature_grid,norm='l2'),minmax_scaling(feature_grid),maxabs_scaling(feature_grid),standard_scaling(feature_grid)]
        # feature_grid = hstack(rescaling_list)

        X_trainS.append(feature_grid[:-1])
        X_test_S.append(feature_grid[-1:])
        Y_trainS.append(fancy(Y,None,mapping_index[f]))

    return X_trainS,Y_trainS,X_test_S


# code backup..
# nclf = Ridge(alpha=2,fit_intercept=True,bias_no_penalty=True)
# nclf = Dynamic_KNN_Regressor(k=3,verbose=False)
# nclf = grid_search_cv(KNN_Regressor,{'k':[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],'verbose':[True]},nx,ny,verbose=True,is_shuffle=False,scoring='loss')

def merge(ecs_logs,flavors_config,flavors_unique,training_start_time,training_end_time,predict_start_time,predict_end_time,verbose = True):
    predict = {}.fromkeys(flavors_unique)
    for f in flavors_unique:
        predict[f] = 0
    virtual_machine_sum = 0
    mapping_index = get_flavors_unique_mapping(flavors_unique)

    # 1.feature generating
    X_trainS_raw,Y_trainS_raw,X_testS = features_building(ecs_logs,flavors_config,flavors_unique,training_start_time,training_end_time,predict_start_time,predict_end_time)
    
    X_trainS = fancy(X_trainS_raw,None,(0,-1),None)
    Y_trainS = fancy(Y_trainS_raw,None,(0,-1))

    X_valS = fancy(X_trainS_raw,None,(-1,),None)
    Y_valS = fancy(Y_trainS_raw,None,(-1,))
    

    # nx = vstack(X_trainS)
    # ny = vstack(Y_trainS)
    # nclf = KNN_Regressor(k=3)

    
    # clf clustering 

    # 1. trainning process
    clfs = []
    for f in flavors_unique:
        X = X_trainS[mapping_index[f]]
        y = Y_trainS[mapping_index[f]]
        X_test = X_testS[mapping_index[f]]
        # clf = Ridge(alpha=2,fit_intercept=True,bias_no_penalty=False)
        # clf = Ridge(alpha=1,fit_intercept=True,bias_no_penalty=False)
        # clf = grid_search_cv(Ridge,{'alpha':[0.01,0.0001,0.1,1],'fit_intercept':[True,False],'bias_no_penalty':[True,False]},X,y,verbose=True,is_shuffle=False,scoring='score')
        # clf = bagging_estimator(Ridge,{'alpha':3,"fit_intercept":True,"bias_no_penalty":False},max_clf=10)
        clf = KNN_Regressor(k=10,verbose=True)
        # clf = grid_search_cv(KNN_Regressor,{'k':[3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],'verbose':[False]},X,y,verbose=False,is_shuffle=False,scoring='loss')
        # clf = Dynamic_KNN_Regressor(k=3,verbose=False)
        clf.fit(X,y)
        clfs.append(clf)

    val_y = []
    val_y_ = []
    # 2.validation process
    for f in flavors_unique:
        X = X_valS[mapping_index[f]]
        y = Y_valS[mapping_index[f]]
        clf = clfs[mapping_index[f]]
        val_y_.append(clf.predict(X))
        val_y.append(y)

    if verbose:
        print('\n')
        print('###############################################')
        print('validation score-->',official_score(val_y,val_y_))


    test_prediction = []
    # 3.retraining
    for f in flavors_unique:
        X = X_trainS_raw[mapping_index[f]]
        y = Y_trainS_raw[mapping_index[f]]
        X_test = X_testS[mapping_index[f]]
        clf = clfs[mapping_index[f]]
        clf.fit(X,y)
        p = clf.predict(X_test)
        test_prediction.append(p)

    result = matrix_transpose(test_prediction)[0]

    if verbose:
        print('predict-->',result)

    result = [0 if r<0 else r for r in result]
    for f in flavors_unique:
        p = result[mapping_index[f]]
        predict[f] = int(round(p))
        virtual_machine_sum += int(round(p))
    return predict,virtual_machine_sum



# build output lines
def predict_vm(ecs_lines,input_lines):
    if input_lines is None or ecs_lines is None:
        return []

    machine_config,flavors_config,flavors_unique,optimized,predict_start_time,predict_end_time = parse_input_lines(input_lines)
    ecs_logs,training_start_time,training_end_time = parse_ecs_lines(ecs_lines,flavors_unique)

    predict,virtual_machine_sum = merge(ecs_logs,flavors_config,flavors_unique,training_start_time,training_end_time,predict_start_time,predict_end_time)

    result = []
    result.append('{}'.format(virtual_machine_sum))
    for i in flavors_unique:
        result.append('flavor{} {}'.format(i,predict[i]))

    result.append('') # output '\n'

    # backpack_list,entity_machine_sum = backpack(machine_config,flavors,flavors_unique,predict)
    backpack_list,entity_machine_sum = backpack_random_k_times(machine_config,flavors_config,flavors_unique,predict,optimized,k=1000)
    
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
