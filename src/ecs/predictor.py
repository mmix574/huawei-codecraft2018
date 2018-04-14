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
def features_building(ecs_logs,flavors_config,flavors_unique,training_start_time,training_end_time,predict_start_time,predict_end_time):
    mapping_index = get_flavors_unique_mapping(flavors_unique)
    predict_days = (predict_end_time-predict_start_time).days

    sample = resampling(ecs_logs,flavors_unique,training_start_time,predict_start_time,frequency=predict_days,strike=1,skip=0)

    def outlier_handling(sample,method='mean',max_sigma=3):
        assert(method=='mean' or method=='zero' or method=='dynamic')
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
                    elif method=='dynamic':
                        sample[i][j] = (sample[i][j] + mean_[j])/2.0
                        
        return sample

    sample = outlier_handling(sample,method='mean',max_sigma=3)
    # sample = exponential_smoothing(sample,alpha=0.1)

    Ys = sample[1:]

    def flavor_clustering(sample,k=3,variance_threshold=None):
        corrcoef_sample = corrcoef(sample)
        clustering_paths = []
        for i in range(shape(sample)[1]):
            col = corrcoef_sample[i]
            col_index_sorted = argsort(col)[::-1]
            if variance_threshold!=None:
                col_index_sorted = col_index_sorted[1:]
                index = [i  for i in col_index_sorted if col[i]>variance_threshold]
            else:
                index = col_index_sorted[1:k+1]
            clustering_paths.append(index)
        return clustering_paths,corrcoef_sample


    # adjustable # 1
    variance_threshold = 0.6 #76.234

    clustering_paths,coef_sample = flavor_clustering(sample,variance_threshold=variance_threshold)

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


    # def get_rate_X(sample,j):
    #     sum_row = sum(sample,axis=1)
    #     A = [sample[i][j]/float(sum_row[i]) if sum_row[i]!=0 else 0 for i in range(shape(sample)[0])]
    #     return A

    # def get_cpu_rate_X(sample,i):
    #     cpu_config,mem_config = get_machine_config(flavors_unique)
    #     sample_copy = matrix_copy(sample)
    #     for i in range(shape(sample_copy)[0]):
    #         for j in range(shape(sample_copy)[1]):
    #             sample_copy[i][j] *= cpu_config[j]

    #     sample = sample_copy
    #     sum_row = sum(sample,axis=1)
    #     A = [sample[i][j]/float(sum_row[i]) if sum_row[i]!=0 else 0 for i in range(shape(sample)[0])]
    #     return A

    # def get_men_rate_X(sample,i):
    #     cpu_config,mem_config = get_machine_config(flavors_unique)
    #     sample_copy = matrix_copy(sample)
    #     for i in range(shape(sample_copy)[0]):
    #         for j in range(shape(sample_copy)[1]):
    #             sample_copy[i][j] *= mem_config[j]

    #     sample = sample_copy
    #     sum_row = sum(sample,axis=1)
    #     A = [sample[i][j]/float(sum_row[i]) if sum_row[i]!=0 else 0 for i in range(shape(sample)[0])]
    #     return A

    X_trainS,Y_trainS,X_test_S = [],[],[]


    # adjustable # 2 
    col_count = 5 # n_feature

    for f in flavors_unique:
        X = get_feature_grid(sample,mapping_index[f],col_count=col_count,fill_na='mean',max_na_rate=1,with_test=True)
        X_test = X[-1:]
        X = X[:-1]
        y = fancy(Ys,None,(mapping_index[f],mapping_index[f]+1))


        clustering = True
        # 1.data clustering 
        if clustering:
            print(clustering_paths[mapping_index[f]])
            # improve weights of X and y
            X.extend(X)
            y.extend(y)

            for cluster_index in clustering_paths[mapping_index[f]]:
                X_cluster = get_feature_grid(sample,mapping_index[f],col_count=col_count,fill_na='mean',max_na_rate=1,with_test=False)
                y_cluster = fancy(Ys,None,(cluster_index,cluster_index+1))
                w =  coef_sample[mapping_index[f]][cluster_index]

                # important
                X_cluster = apply(X_cluster,lambda x:x*w)
                y_cluster = apply(y_cluster,lambda x:x*w)

                X.extend(X_cluster)
                y.extend(y_cluster)

        # do not delete
        X.extend(X_test)


        # --------------------------------------------------------- #


        # --------------------------------------------------------- #

        def multi_exponential_smoothing(A,list_of_alpha):
            R = A
            for a in list_of_alpha:
                R = exponential_smoothing(R,alpha=a)
            return R

        #adjustable #3 smoothing degree 
        # 77.291 3
        #	77.405 no.63
        depth = 3
        #adjustable #4 smoothing weights
        # base = [0.3,0.5,0.7,0.8] # 3.0.6,0.7,0.8 77.163
        # base = [0.1,0.3,0.5] # 3.0.6,0.7,0.8 77.163
        base = [0.6,0.7,0.8]

        # depth = 3
        # base = [0.7,0.8,0.9]


        alphas = [[ base[i]  for _ in range(depth)]for i in range(len(base))]

        X_data_list = [multi_exponential_smoothing(X[:-1],a) for a in alphas]
        Y_data_list = [multi_exponential_smoothing(y,a) for a in alphas]
        
        X_data_list.extend([X])
        Y_data_list.extend([y])
        X = vstack(X_data_list)
        y = vstack(Y_data_list)

        # # --------------------------------------------------------- #

        add_list= [X]
        # add_list.extend([X_diff(X)])
        add_list.extend([apply(X,lambda x:math.log1p(x))]) # important
        X = hstack(add_list)


        # -----------------------------------------------------------#

        y = flatten(y)
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

    R = []
    X_trainS_raw,Y_trainS_raw,X_testS = features_building(ecs_logs,flavors_config,flavors_unique,training_start_time,training_end_time,predict_start_time,predict_end_time)

    X_trainS = fancy(X_trainS_raw,None,(0,-1),None)
    Y_trainS = fancy(Y_trainS_raw,None,(0,-1))

    X_valS = fancy(X_trainS_raw,None,(-1,),None)
    Y_valS = fancy(Y_trainS_raw,None,(-1,))

    #adjustable #5 Ridge Regression alpha
    clf = Ridge(alpha=1)

    test = []
    train = []
    val = []
    for i in range(len(flavors_unique)):    
        X = X_trainS[i]
        y = Y_trainS[i]
        clf.fit(X,y)
        train.append(clf.predict(X))
        val.append(clf.predict(X_valS[i]))
        test.append(clf.predict(X_testS[i]))

    # print("shape(train)",shape(train))

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
    backpack_list,entity_machine_sum = backpack_random_k_times(machine_config,flavors_config,flavors_unique,predict,optimized,k=10000)
    

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
