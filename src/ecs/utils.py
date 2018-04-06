from datetime import datetime
import math
import re

from linalg.common import dim,shape,reshape,zeros,mean,multiply,square,sqrt,dot
from linalg.matrix import matrix_matmul,matrix_transpose,hstack,shift
from linalg.vector import count_nonezero

def parse_input_lines(input_lines):
    input_lines = [x.strip() for x in input_lines]

    machine_config = {'cpu':None,'mem':None,'disk':None}
    flavors_number = 0
    flavors = {}
    optimized = 'CPU'
    predict_times = [] 

    flavors_unique = []

    seg = 1
    for line in input_lines:
        if line == '':
            seg += 1
            continue
        if seg == 1:
            machine_config['cpu'] = int(line.split(' ')[0])
            machine_config['mem'] = int(line.split(' ')[1]) * 1024
            machine_config['disk'] = int(line.split(' ')[2])
        elif seg == 2:
            if not flavors_number:
                flavors_number = int(line)
                continue                
            f,core,mem = line.split(' ')
            f,core,mem = int(f[f.find('r')+1:]),int(core),int(mem)
            # core = int(core)
            # mem = int(mem)
            flavors[f]=(core,mem)
            flavors_unique.append(f)
        elif seg == 3:
            optimized = line.strip()
        elif seg == 4:
            predict_times.append(line.strip())

    predict_start_time = datetime.strptime(predict_times[0], "%Y-%m-%d %H:%M:%S")
    predict_end_time = datetime.strptime(predict_times[1], "%Y-%m-%d %H:%M:%S")

    # sorted flavors_unique @date 03-15
    flavors_unique = sorted(flavors_unique)

    return machine_config,flavors_number,flavors,flavors_unique,optimized,predict_start_time,predict_end_time

def parse_ecs_lines(ecs_lines):
    ecs_logs = []
    if(len(ecs_lines)==0):
        return ecs_logs,None,None
        
    for line in ecs_lines:
        _uuid,f,t = line.split('\t')
        f = int(f[f.find('r')+1:])
        t = datetime.strptime(t.strip(), "%Y-%m-%d %H:%M:%S")
        ecs_logs.append((f,t))

    training_start_time = ecs_logs[0][1]
    training_end_time = ecs_logs[len(ecs_logs)-1][1]
    return ecs_logs,training_start_time,training_end_time

# column vector corrcoef
def corrcoef(A):
    assert(dim(A)==2)
    m,n = shape(A)
    def _corr(A,i,j):
        assert(dim(A)==2)
        m,n = shape(A)
        A_T = matrix_transpose(A)
        
        X,Y = A_T[i],A_T[j] # X,Y = col(A,i),col(A,j)

        mean_X,mean_Y = mean(X),mean(Y)
        X_ = [k-mean_X for k in X]
        Y_ = [k-mean_Y for k in Y]
        numerator = mean(multiply(X_,Y_))
        # print(sqrt(mean(square(X_))))

        denominator = sqrt(mean(square(X_)))*sqrt(mean(square(Y_)))
        if denominator == 0:
            return 0
        else:
            r = (numerator)/(denominator)
            return r

    R = zeros((n,n))
    for i in range(n):
        for j in range(n):
            if i==j:
                R[i][j] = 1
            elif i>j:
                R[i][j] = R[j][i]
            else:
                R[i][j] = _corr(A,i,j)
    return R

# def arange(start,end,total_points):
#     h = (end-start)/total_points
#     return [start + i*h for i in range(total_points)]


from linalg.common import (dim, dot, mean, multiply, reshape, shape, sqrt,
                           square, zeros,plus,sum,minus)
# 2018-04-02
# vector and matrix supportted
def l2_loss(y,y_):
    # if dim(y) == 2:
    #     return mean(sqrt(mean(square(minus(y,y_)),axis=0)))
    # else:
    #     return sqrt(mean(square(minus(y,y_))))

    def _score_calc(y,y_):
        numerator = sqrt(mean(square(minus(y,y_))))
        return numerator

    if dim(y) == 1:
        return _score_calc(y,y_)
    else:
        return mean([_score_calc(y[i],y_[i]) for i in range(len(y))])

# 2018-04-02
# vector and matrix supportted
def official_score(y,y_):
    def _score_calc(y,y_):
        numerator = sqrt(mean(square(minus(y,y_))))
        denominator = sqrt(mean(square(y))) + sqrt(mean(square(y_)))
        if denominator==0:
            return 0
        else:
            return 1-(numerator/float(denominator))
    if dim(y) == 1:
        return _score_calc(y,y_)
    else:
        return mean([_score_calc(y[i],y_[i]) for i in range(len(y))])


def get_flavors_unique_mapping(flavors_unique):
    mapping_index = {}.fromkeys(flavors_unique)
    c = 0
    for f in flavors_unique:
        mapping_index[f] = c
        c+=1
    return mapping_index