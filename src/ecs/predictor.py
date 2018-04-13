import math
import random
import re
from datetime import datetime, timedelta

from backpack import backpack_random_k_times
from ensemble import bagging_estimator
from learn.knn import Dynamic_KNN_Regressor, KNN_Regressor
from learn.lasso import Lasso
from learn.linear_model import LinearRegression
from learn.ridge import Ridge
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
from utils import (get_flavors_unique_mapping, get_machine_config,
                   parse_ecs_lines, parse_input_lines)

from preprocessing import exponential_smoothing

# add @2018-04-10
# refactoring, do one thing.
def resampling(ecs_logs,flavors_unique,training_start_time,predict_start_time,frequency=7,strike=1,skip=0):
    predict_start_time = predict_start_time-timedelta(days=skip)
    days_total = (predict_start_time-training_start_time).days

    sample_length = ((days_total-frequency)/strike) + 1
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
    # [         ...               ]
    # [         ...               ]
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
    

# fix griding bug @2018-04-12
def features_building(ecs_logs,flavors_config,flavors_unique,training_start_time,training_end_time,predict_start_time,predict_end_time,variance_threshold=0.5):
    mapping_index = get_flavors_unique_mapping(flavors_unique)
    predict_days = (predict_end_time-predict_start_time).days

    sample = resampling(ecs_logs,flavors_unique,training_start_time,predict_start_time,frequency=predict_days,strike=3,skip=0)

    def outlier_handling(sample,method='mean',max_sigma=3):
        assert(method=='mean' or method=='zero')
        sample = matrix_copy(sample)
        std_ = stdev(sample)
        mean_ = mean(sample,axis=1)
        for i in range(shape(sample)[0]):
            for j in range(shape(sample)[1]):
               if sample[i][j]-mean_[j] >max_sigma*std_[j]:
                    if method=='mean':
                        sample[i][j] = mean_[j]
                    elif method=='zero':
                        sample[i][j] = 0
        return sample

    # sample = outlier_handling(sample,method='zero',max_sigma=5)
    sample = outlier_handling(sample,method='mean',max_sigma=3)
    # sample = exponential_smoothing(sample,alpha=0.1)
    
    Ys = sample[1:]

    def get_feature_grid(sample,i,fill_na='mean',max_na_rate=1,col_count=None,with_test=True):
        assert(fill_na=='mean' or fill_na=='zero')
        col = fancy(sample,None,i)
        R = []
        for j in range(len(col)):
            left = [None for _ in range(len(col)-j)]
            right = col[:j]
            r = []
            r.extend(left)
            r.extend(right)
            R.append(r)

        def _mean_with_none(A):
            if len(A)==0:
                return 0
            else:
                count = 0
                for i in range(len(A)):
                    if A[i]!=None:
                        count+=A[i]
                return count/float(len(A))
        
        means = []
        for j in range(shape(R)[1]):
            means.append(_mean_with_none(fancy(R,None,j)))
        
        width = int((1-max_na_rate) * shape(R)[1])
        R = fancy(R,None,(width,))
        for _ in range(shape(R)[0]):
            for j in range(shape(R)[1]):
                    if R[_][j]==None:
                        if fill_na=='mean':
                            R[_][j] = means[j]
                        elif fill_na=='zero':
                            R[_][j]=0
        if with_test:
            if col_count!=None:
                return fancy(R,None,(-col_count,))
            else:
                return R
        else:
            if col_count!=None:
                return fancy(R,(0,-1),(-col_count,))
            else:            
                return R[:-1]


    def get_rate_X(sample,i):
        sum_row = sum(sample,axis=1)
        R = [ multiply(sample[i],1/float(sum_row[i])) if sum_row[i] !=0 else 1  for i in range(len(sample))]
        return fancy(R,None,(i,i+1))

    def get_cpu_rate_X(sample,i):
        cpu_config,mem_config = get_machine_config(flavors_unique)
        sample_copy = matrix_copy(sample)
        for i in range(shape(sample_copy)[0]):
            for j in range(shape(sample_copy)[1]):
                sample_copy[i][j] *= cpu_config[j]
        sum_row = sum(sample_copy,axis=1)
        R = [ multiply(sample_copy[i],1/float(sum_row[i])) if sum_row[i] !=0 else 1  for i in range(len(sample_copy))]
        return fancy(R,None,(i,i+1))

    def get_men_rate_X(sample,i):
        cpu_config,mem_config = get_machine_config(flavors_unique)
        sample_copy = matrix_copy(sample)
        for i in range(shape(sample_copy)[0]):
            for j in range(shape(sample_copy)[1]):
                sample_copy[i][j] *= mem_config[j]
        sum_row = sum(sample_copy,axis=1)
        R = [ multiply(sample_copy[i],1/float(sum_row[i])) if sum_row[i] !=0 else 1  for i in range(len(sample_copy))]
        return fancy(R,None,(i,i+1))

    X_trainS,Y_trainS,X_test_S = [],[],[]

    for f in flavors_unique:
        X = get_feature_grid(sample,mapping_index[f],col_count=4,fill_na='mean',max_na_rate=1,with_test=True)
        X_test = X[-1:]
        X = X[:-1]
        y = fancy(Ys,None,mapping_index[f])

        # X = vstack([X,X[:-1]])
        # y = vstack([y,y[1:]])

        X.extend(X_test)

        # X_rate = get_rate_X(sample,mapping_index[f])
        # X_cpu_rate = get_cpu_rate_X(sample,mapping_index[f])
        # X_mem_rate = get_men_rate_X(sample,mapping_index[f])
        
        add_list= [X]
        # add_list.extend([X_rate,X_cpu_rate,X_mem_rate])
        add_list.extend([square(X)])
        X = hstack(add_list)

        X = normalize(X,y=y,norm='l1')
        assert(shape(X)[0]==shape(y)[0]+1)
        X_trainS.append(X[:-1])
        X_test_S.append(X[-1:])
        Y_trainS.append(y)

    return X_trainS,Y_trainS,X_test_S


def merge(ecs_logs,flavors_config,flavors_unique,training_start_time,training_end_time,predict_start_time,predict_end_time):
    predict = {}.fromkeys(flavors_unique)
    for f in flavors_unique:
        predict[f] = 0
    virtual_machine_sum = 0
    mapping_index = get_flavors_unique_mapping(flavors_unique)

    # clf = Ridge(alpha=0.1,fit_intercept=True)
    clf = Ridge(alpha=1,fit_intercept=True)

    R = []
    X_trainS_raw,Y_trainS_raw,X_testS = features_building(ecs_logs,flavors_config,flavors_unique,training_start_time,training_end_time,predict_start_time,predict_end_time)

    X_trainS = fancy(X_trainS_raw,None,(0,-1),None)
    Y_trainS = fancy(Y_trainS_raw,None,(0,-1))

    X_valS = fancy(X_trainS_raw,None,(-1,),None)
    Y_valS = fancy(Y_trainS_raw,None,(-1,))

    test = []
    train = []
    val = []
    for i in range(len(flavors_unique)):    
        # X = X_trainS[i]
        # y = Y_trainS[i]
        X = X_trainS[i]
        y = Y_trainS[i]
        # clf = grid_search_cv(Ridge,{'alpha':[0.1,0.2,0.3,0.4,0.5,0.8,1,1.5,2,3,4]},X,y,is_shuffle=False,verbose=True,random_state=41,cv='full',scoring='score')
        clf.fit(X,y)
        train.append(clf.predict(X))
        val.append(clf.predict(X_valS[i]))
        test.append(clf.predict(X_testS[i]))

    train = matrix_transpose(train)
    Y_trainS = matrix_transpose(Y_trainS)
    R.extend(test)

    print("training_score-->",official_score(train,Y_trainS))
    val = matrix_transpose(val)
    Y_valS = matrix_transpose(Y_valS)
    print("validation_score-->",official_score(val,Y_valS))
    
    result = flatten(R)
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
