from datetime import datetime
import random
import math
import re

from linalg.common import dim,shape,reshape,zeros
from linalg.matrix import matrix_mutmul,matrix_transpose
from linalg.vector import count_nonezero,mean,squrt,minus,plus,root,multiply

def parse_input_lines(input_lines):
    input_lines = [x.strip() for x in input_lines]

    machine_config = {'cpu':None,'mem':None,'disk':None}
    flavors_number = 0
    flavors = {}
    optimized = 'CPU'
    predict_times = [] 

    flavors_unique = []

    sf = 1
    for line in input_lines:
        if line == '':
            sf += 1
            continue
        if sf == 1:
            machine_config['cpu'] = int(line.split(' ')[0])
            machine_config['mem'] = int(line.split(' ')[1]) * 1024
            machine_config['disk'] = int(line.split(' ')[2])
        elif sf == 2:
            if not flavors_number:
                flavors_number = int(line)
                continue                
            f,core,mem = line.split(' ')
            f = int(f[f.find('r')+1:])
            core = int(core)
            mem = int(mem)
            flavors[f]=(core,mem)
            flavors_unique.append(f)
        elif sf == 3:
            optimized = line.strip()
        elif sf == 4:
            predict_times.append(line.strip())

    predict_start_time = datetime.strptime(predict_times[0], "%Y-%m-%d %H:%M:%S")
    predict_end_time = datetime.strptime(predict_times[1], "%Y-%m-%d %H:%M:%S")

    # sorted flavors_unique @date 03-15
    flavors_unique = sorted(flavors_unique)

    assert(machine_config['cpu']!=None) #passed
    assert(flavors_number!=0) #passed
    assert(len(flavors)!=0) #passed
    assert(len(flavors_unique)!=0) #passed

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


def _corr(A,i,j):
    assert(dim(A)==2)
    m,n = shape(A)
    A_T = matrix_transpose(A)
    # X,Y = col(A,i),col(A,j)
    X,Y = A_T[i],A_T[j]

    mean_X,mean_Y = mean(X),mean(Y)
    X_ = [k-mean_X for k in X]
    Y_ = [k-mean_Y for k in Y]
    numerator = mean(multiply(X_,Y_))
    denominator = root(mean(squrt(X_)))*root(mean(squrt(Y_)))
    if denominator == 0:
        return 0
    else:
        r = (numerator)/(denominator)
        return r

def correlation(A):
    assert(dim(A)==2)
    m,n = shape(A)
    global R
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

# sample[0] is latest frequency slicing 
def resample(ecs_logs,flavors_unique,training_start_time,predict_start_time,frequency='7d'):
    # align to right
    assert(frequency[len(frequency)-1]=='d')
    span = int(frequency[:-1])
    training_days = ((predict_start_time - training_start_time).days) +1 
    max_sample_length = training_days//span

    sample = zeros((max_sample_length,len(flavors_unique)))
    mapping_index = {}.fromkeys(flavors_unique)
    
    c = 0
    for f in flavors_unique:
        mapping_index[f] = c
        c+=1

    for flavor,ecs_time in ecs_logs:
        ith = (predict_start_time - ecs_time).days//span
        if ith >= len(sample):
            # filter some ecs_logs not enough for this frequency slicing
            continue
        if flavor not in flavors_unique:
            # filter some flavor not in the given flavors_unique
            continue
        # add count 
        sample[ith][mapping_index[flavor]] += 1

    return sample,mapping_index    


def _corr_prior(flavors_unique):
    _prior_mapping_index = {1:0,2:1,3:2,4:3,5:4,6:5,7:6,8:7,9:8,10:9,11:10,12:11,13:12,14:13,15:14,17:15,18:16,21:17,22:18,23:19}
    _prior_correlation_filter_0 = [[  2.52813853e+00,   3.12121212e+00,   9.43722944e-01,
          2.59740260e-01,   1.44502165e+01,   5.90476190e+00,
          1.57575758e+00,   7.95238095e+00,   2.54112554e+00,
          8.91774892e-01,   1.91774892e+00,   1.31601732e+00,
          4.08225108e+00,  -1.21212121e-01,  -2.15584416e+00,
          3.29004329e-01,  -1.16883117e-01,  -8.22510823e-02,
          3.98268398e-01,  -1.29870130e-01],
       [  3.12121212e+00,   3.51082251e+01,   2.33766234e-01,
         -6.06060606e-02,   3.13679654e+01,   1.04761905e+01,
          7.07359307e+00,   2.90000000e+01,   1.79610390e+01,
         -5.54112554e-01,   1.03030303e+00,  -4.19913420e+00,
         -3.83982684e+00,   3.46320346e+00,  -4.88744589e+00,
          7.35930736e-01,  -4.15584416e-01,  -3.50649351e-01,
          3.89610390e-01,  -3.98268398e-01],
       [  9.43722944e-01,   2.33766234e-01,   3.23160173e+00,
          9.95670996e-02,   6.57575758e+00,   3.83333333e+00,
         -1.75324675e-01,   3.59523810e+00,   1.15584416e+00,
          3.59307359e-01,   6.88311688e-01,   1.39177489e+00,
          1.71645022e+00,   4.56709957e-01,  -1.11688312e+00,
         -2.77056277e-01,   4.32900433e-02,  -2.59740260e-02,
         -3.67965368e-01,   2.16450216e-02],
       [  2.59740260e-01,  -6.06060606e-02,   9.95670996e-02,
          7.27272727e-01,   1.06060606e+00,   1.19047619e+00,
         -6.92640693e-01,   3.19047619e+00,  -3.41991342e-01,
          1.73160173e-01,   1.11255411e+00,   8.18181818e-01,
          1.41125541e+00,   1.58441558e+00,   1.07792208e+00,
          2.64069264e-01,   3.46320346e-02,   1.12554113e-01,
         -1.03896104e-01,  -7.79220779e-02],
       [  1.44502165e+01,   3.13679654e+01,   6.57575758e+00,
          1.06060606e+00,   1.64346320e+02,   5.47142857e+01,
          8.73593074e+00,   1.05333333e+02,   4.34199134e+01,
          4.55411255e+00,   5.92207792e+00,   1.06753247e+01,
          8.50649351e+00,  -2.07012987e+01,   5.22077922e+00,
         -2.11688312e+00,  -2.51082251e-01,   1.60173160e-01,
         -2.58008658e+00,  -9.35064935e-01],
       [  5.90476190e+00,   1.04761905e+01,   3.83333333e+00,
          1.19047619e+00,   5.47142857e+01,   3.62619048e+01,
          3.88095238e+00,   3.20238095e+01,   1.59523810e+01,
         -8.09523810e-01,   1.27619048e+01,   9.97619048e+00,
         -3.09523810e-01,  -6.88095238e+00,  -3.95238095e+00,
         -5.23809524e-01,   9.52380952e-02,  -3.33333333e-01,
         -4.76190476e-01,  -3.33333333e-01],
       [  1.57575758e+00,   7.07359307e+00,  -1.75324675e-01,
         -6.92640693e-01,   8.73593074e+00,   3.88095238e+00,
          8.40909091e+00,   6.21428571e+00,   1.13506494e+01,
          3.20346320e-01,  -3.46320346e-02,   2.43506494e+00,
         -1.94155844e+00,  -3.66883117e+00,  -9.65367965e-01,
         -5.19480519e-02,   3.59307359e-01,  -1.29870130e-01,
          2.68398268e+00,  -1.77489177e-01],
       [  7.95238095e+00,   2.90000000e+01,   3.59523810e+00,
          3.19047619e+00,   1.05333333e+02,   3.20238095e+01,
          6.21428571e+00,   2.35880952e+02,   2.92857143e+01,
          6.61904762e+00,   1.87142857e+01,  -5.64285714e+00,
         -3.88095238e+00,  -3.54047619e+01,  -9.85714286e+00,
         -3.14285714e+00,  -9.04761905e-01,   1.23809524e+00,
         -7.61904762e-01,  -1.28571429e+00],
       [  2.54112554e+00,   1.79610390e+01,   1.15584416e+00,
         -3.41991342e-01,   4.34199134e+01,   1.59523810e+01,
          1.13506494e+01,   2.92857143e+01,   4.04502165e+01,
          3.23376623e+00,  -4.04329004e+00,   4.50216450e+00,
         -3.81385281e+00,  -5.38961039e+00,   6.04329004e+00,
         -1.25541126e+00,  -8.65800866e-02,   1.94805195e-01,
          2.12121212e-01,  -2.33766234e-01],
       [  8.91774892e-01,  -5.54112554e-01,   3.59307359e-01,
          1.73160173e-01,   4.55411255e+00,  -8.09523810e-01,
          3.20346320e-01,   6.61904762e+00,   3.23376623e+00,
          3.89610390e+00,   7.70562771e-01,  -2.04329004e+00,
          8.18181818e+00,   8.87445887e-01,  -8.65800866e-01,
          1.29870130e-02,  -1.73160173e-01,  -3.89610390e-02,
         -3.85281385e-01,  -8.65800866e-02],
       [  1.91774892e+00,   1.03030303e+00,   6.88311688e-01,
          1.11255411e+00,   5.92207792e+00,   1.27619048e+01,
         -3.46320346e-02,   1.87142857e+01,  -4.04329004e+00,
          7.70562771e-01,   1.71341991e+01,   3.42424242e+00,
          4.38961039e+00,   3.54112554e+00,  -6.70562771e+00,
          2.72727273e-01,   1.25541126e-01,  -3.41991342e-01,
          3.85281385e-01,   8.65800866e-02],
       [  1.31601732e+00,  -4.19913420e+00,   1.39177489e+00,
          8.18181818e-01,   1.06753247e+01,   9.97619048e+00,
          2.43506494e+00,  -5.64285714e+00,   4.50216450e+00,
         -2.04329004e+00,   3.42424242e+00,   1.84264069e+01,
         -1.25757576e+00,   2.17532468e+00,   1.29090909e+01,
         -1.25541126e-01,   5.62770563e-01,   3.76623377e-01,
          1.02597403e+00,   6.62337662e-01],
       [  4.08225108e+00,  -3.83982684e+00,   1.71645022e+00,
          1.41125541e+00,   8.50649351e+00,  -3.09523810e-01,
         -1.94155844e+00,  -3.88095238e+00,  -3.81385281e+00,
          8.18181818e+00,   4.38961039e+00,  -1.25757576e+00,
          2.89199134e+01,   4.00649351e+00,  -2.81818182e+00,
         -9.39393939e-01,  -2.20779221e-01,  -1.34199134e-01,
         -3.85281385e-01,  -1.34199134e-01],
       [ -1.21212121e-01,   3.46320346e+00,   4.56709957e-01,
          1.58441558e+00,  -2.07012987e+01,  -6.88095238e+00,
         -3.66883117e+00,  -3.54047619e+01,  -5.38961039e+00,
          8.87445887e-01,   3.54112554e+00,   2.17532468e+00,
          4.00649351e+00,   5.83225108e+01,  -2.16017316e+00,
          9.59740260e+00,  -1.55844156e-01,  -3.03030303e-02,
         -1.19913420e+00,   1.39826840e+00],
       [ -2.15584416e+00,  -4.88744589e+00,  -1.11688312e+00,
          1.07792208e+00,   5.22077922e+00,  -3.95238095e+00,
         -9.65367965e-01,  -9.85714286e+00,   6.04329004e+00,
         -8.65800866e-01,  -6.70562771e+00,   1.29090909e+01,
         -2.81818182e+00,  -2.16017316e+00,   2.81818182e+01,
         -9.39393939e-01,   9.22077922e-01,   1.00865801e+00,
         -8.13852814e-01,  -8.65800866e-02],
       [  3.29004329e-01,   7.35930736e-01,  -2.77056277e-01,
          2.64069264e-01,  -2.11688312e+00,  -5.23809524e-01,
         -5.19480519e-02,  -3.14285714e+00,  -1.25541126e+00,
          1.29870130e-02,   2.72727273e-01,  -1.25541126e-01,
         -9.39393939e-01,   9.59740260e+00,  -9.39393939e-01,
          3.19480519e+00,  -1.21212121e-01,  -6.06060606e-02,
          7.79220779e-02,   3.46320346e-02],
       [ -1.16883117e-01,  -4.15584416e-01,   4.32900433e-02,
          3.46320346e-02,  -2.51082251e-01,   9.52380952e-02,
          3.59307359e-01,  -9.04761905e-01,  -8.65800866e-02,
         -1.73160173e-01,   1.25541126e-01,   5.62770563e-01,
         -2.20779221e-01,  -1.55844156e-01,   9.22077922e-01,
         -1.21212121e-01,   2.51082251e-01,   3.03030303e-02,
          2.94372294e-01,  -1.73160173e-02],
       [ -8.22510823e-02,  -3.50649351e-01,  -2.59740260e-02,
          1.12554113e-01,   1.60173160e-01,  -3.33333333e-01,
         -1.29870130e-01,   1.23809524e+00,   1.94805195e-01,
         -3.89610390e-02,  -3.41991342e-01,   3.76623377e-01,
         -1.34199134e-01,  -3.03030303e-02,   1.00865801e+00,
         -6.06060606e-02,   3.03030303e-02,   8.65800866e-02,
         -4.32900433e-02,  -8.65800866e-03],
       [  3.98268398e-01,   3.89610390e-01,  -3.67965368e-01,
         -1.03896104e-01,  -2.58008658e+00,  -4.76190476e-01,
          2.68398268e+00,  -7.61904762e-01,   2.12121212e-01,
         -3.85281385e-01,   3.85281385e-01,   1.02597403e+00,
         -3.85281385e-01,  -1.19913420e+00,  -8.13852814e-01,
          7.79220779e-02,   2.94372294e-01,  -4.32900433e-02,
          1.68831169e+00,  -4.32900433e-02],
       [ -1.29870130e-01,  -3.98268398e-01,   2.16450216e-02,
         -7.79220779e-02,  -9.35064935e-01,  -3.33333333e-01,
         -1.77489177e-01,  -1.28571429e+00,  -2.33766234e-01,
         -8.65800866e-02,   8.65800866e-02,   6.62337662e-01,
         -1.34199134e-01,   1.39826840e+00,  -8.65800866e-02,
          3.46320346e-02,  -1.73160173e-02,  -8.65800866e-03,
         -4.32900433e-02,   1.81818182e-01]]
    # _prior_correlation_filter_0 = [[ 1.        ,  0.33129769,  0.33016814,  0.19155354,  0.70891461,
    #      0.61670384,  0.34175466,  0.32564942,  0.25128425,  0.28414469,
    #      0.29137978,  0.19281501,  0.47742019, -0.00998223, -0.25540678,
    #      0.11576546, -0.14670447, -0.17580539,  0.19277428, -0.19155354],
    #    [ 0.33129769,  1.        ,  0.02194666, -0.01199397,  0.41295423,
    #      0.29361189,  0.41168127,  0.31867436,  0.47661378, -0.04737817,
    #      0.04200767, -0.16509541, -0.12050601,  0.07653419, -0.15537925,
    #      0.06948808, -0.13997382, -0.20112201,  0.05060574, -0.15763504],
    #    [ 0.33016814,  0.02194666,  1.        ,  0.06494688,  0.28533621,
    #      0.3541134 , -0.03363254,  0.13021838,  0.10109506,  0.10126093,
    #      0.09250055,  0.1803598 ,  0.17755128,  0.03326699, -0.11703461,
    #     -0.08622573,  0.0480586 , -0.04910451, -0.15753306,  0.02823777],
    #    [ 0.19155354, -0.01199397,  0.06494688,  1.        ,  0.097012  ,
    #      0.23181779, -0.28008176,  0.24359054, -0.06305304,  0.1028689 ,
    #      0.31516689,  0.22350171,  0.30772195,  0.24327773,  0.23809716,
    #      0.17323954,  0.08104409,  0.44854261, -0.09376145, -0.21428571],
    #    [ 0.70891461,  0.41295423,  0.28533621,  0.097012  ,  1.        ,
    #      0.70875386,  0.23499286,  0.53498228,  0.53253547,  0.17997348,
    #      0.11159949,  0.19399064,  0.12338776, -0.21144601,  0.07671333,
    #     -0.09238356, -0.03908661,  0.04246203, -0.15489164, -0.1710579 ],
    #    [ 0.61670384,  0.29361189,  0.3541134 ,  0.23181779,  0.70875386,
    #      1.        ,  0.22224832,  0.34625914,  0.41652376, -0.06810658,
    #      0.51198624,  0.38593938, -0.00955807, -0.14962534, -0.12363715,
    #     -0.04866603,  0.03156294, -0.1881241 , -0.0608596 , -0.12981796],
    #    [ 0.34175466,  0.41168127, -0.03363254, -0.28008176,  0.23499286,
    #      0.22224832,  1.        ,  0.13953086,  0.61544006,  0.05596673,
    #     -0.00288517,  0.1956211 , -0.12450232, -0.16566665, -0.06270957,
    #     -0.01002242,  0.24727692, -0.15220389,  0.71232612, -0.1435419 ],
    #    [ 0.32564942,  0.31867436,  0.13021838,  0.24359054,  0.53498228,
    #      0.34625914,  0.13953086,  1.        ,  0.29981196,  0.21834025,
    #      0.29437093, -0.08559183, -0.04698869, -0.30185414, -0.12089828,
    #     -0.11448696, -0.11756549,  0.27396722, -0.0381793 , -0.19632671],
    #    [ 0.25128425,  0.47661378,  0.10109506, -0.06305304,  0.53253547,
    #      0.41652376,  0.61544006,  0.29981196,  1.        ,  0.25759233,
    #     -0.15358279,  0.16490758, -0.11150774, -0.11096317,  0.17898989,
    #     -0.11043415, -0.02716749,  0.1040953 ,  0.02566832, -0.08619909],
    #    [ 0.28414469, -0.04737817,  0.10126093,  0.1028689 ,  0.17997348,
    #     -0.06810658,  0.05596673,  0.21834025,  0.25759233,  1.        ,
    #      0.09431067, -0.24115398,  0.77078999,  0.05887196, -0.08262629,
    #      0.00368105, -0.17507524, -0.06708204, -0.15022301, -0.1028689 ],
    #    [ 0.29137978,  0.04200767,  0.09250055,  0.31516689,  0.11159949,
    #      0.51198624, -0.00288517,  0.29437093, -0.15358279,  0.09431067,
    #      1.        ,  0.19271356,  0.19719485,  0.1120189 , -0.30515678,
    #      0.03686162,  0.06052658, -0.28078519,  0.0716341 ,  0.04905321],
    #    [ 0.19281501, -0.16509541,  0.1803598 ,  0.22350171,  0.19399064,
    #      0.38593938,  0.1956211 , -0.08559183,  0.16490758, -0.24115398,
    #      0.19271356,  1.        , -0.05447727,  0.06635683,  0.56648815,
    #     -0.01636227,  0.26163934,  0.29817961,  0.18394574,  0.36185991],
    #    [ 0.47742019, -0.12050601,  0.17755128,  0.30772195,  0.12338776,
    #     -0.00955807, -0.12450232, -0.04698869, -0.11150774,  0.77078999,
    #      0.19719485, -0.05447727,  1.        ,  0.09755462, -0.09871565,
    #     -0.09772981, -0.08193165, -0.08480905, -0.05513828, -0.05852381],
    #    [-0.00998223,  0.07653419,  0.03326699,  0.24327773, -0.21144601,
    #     -0.14962534, -0.16566665, -0.30185414, -0.11096317,  0.05887196,
    #      0.1120189 ,  0.06635683,  0.09755462,  1.        , -0.05328267,
    #      0.70309404, -0.04072531, -0.01348525, -0.12084352,  0.42939183],
    #    [-0.25540678, -0.15537925, -0.11703461,  0.23809716,  0.07671333,
    #     -0.12363715, -0.06270957, -0.12089828,  0.17898989, -0.08262629,
    #     -0.30515678,  0.56648815, -0.09871565, -0.05328267,  1.        ,
    #     -0.09900134,  0.34663716,  0.6457292 , -0.11798724, -0.03824854],
    #    [ 0.11576546,  0.06948808, -0.08622573,  0.17323954, -0.09238356,
    #     -0.04866603, -0.01002242, -0.11448696, -0.11043415,  0.00368105,
    #      0.03686162, -0.01636227, -0.09772981,  0.70309404, -0.09900134,
    #      1.        , -0.13533678, -0.11523512,  0.03355151,  0.04543988],
    #    [-0.14670447, -0.13997382,  0.0480586 ,  0.08104409, -0.03908661,
    #      0.03156294,  0.24727692, -0.11756549, -0.02716749, -0.17507524,
    #      0.06052658,  0.26163934, -0.08193165, -0.04072531,  0.34663716,
    #     -0.13533678,  1.        ,  0.20552708,  0.45212926, -0.08104409],
    #    [-0.17580539, -0.20112201, -0.04910451,  0.44854261,  0.04246203,
    #     -0.1881241 , -0.15220389,  0.27396722,  0.1040953 , -0.06708204,
    #     -0.28078519,  0.29817961, -0.08480905, -0.01348525,  0.6457292 ,
    #     -0.11523512,  0.20552708,  1.        , -0.1132277 , -0.06900656],
    #    [ 0.19277428,  0.05060574, -0.15753306, -0.09376145, -0.15489164,
    #     -0.0608596 ,  0.71232612, -0.0381793 ,  0.02566832, -0.15022301,
    #      0.0716341 ,  0.18394574, -0.05513828, -0.12084352, -0.11798724,
    #      0.03355151,  0.45212926, -0.1132277 ,  1.        , -0.07813454],
    #    [-0.19155354, -0.15763504,  0.02823777, -0.21428571, -0.1710579 ,
    #     -0.12981796, -0.1435419 , -0.19632671, -0.08619909, -0.1028689 ,
    #      0.04905321,  0.36185991, -0.05852381,  0.42939183, -0.03824854,
    #      0.04543988, -0.08104409, -0.06900656, -0.07813454,  1.        ]]

    # filtering
    for i in range(len(_prior_correlation_filter_0)):
        for j in range(len(_prior_correlation_filter_0[0])):
            if _prior_correlation_filter_0[i][j]<0:
                _prior_correlation_filter_0[i][j] = 0

    # extract prior correlation base on flavors_unique
    prior_correlation = zeros((len(flavors_unique),len(flavors_unique)))
    prior_mapping_index = {}.fromkeys(flavors_unique)    

    for i in range(len(flavors_unique)):
        prior_mapping_index[flavors_unique[i]] = i

    for i in range(len(flavors_unique)):
        for j in range(len(flavors_unique)):
            prior_correlation[i][j] = _prior_correlation_filter_0[_prior_mapping_index[flavors_unique[i]]][_prior_mapping_index[flavors_unique[j]]]

    # rescaling factor
    for i in range(len(prior_correlation)):
        s = sum([abs(x) for x in prior_correlation[i]])
        s = s if s!=0 else 1
        for j in range(len(prior_correlation[0])):
            prior_correlation[i][j] /= s
    return prior_correlation, prior_mapping_index


# a template, you can modify it according to your methods. 
# def predict_flavors_unique_template(ecs_logs,flavors_unique,training_start_time,training_end_time,predict_start_time,predict_end_time):
#     # modify @ 2018-03-15 
#     predict = {}.fromkeys(flavors_unique)
#     for f in flavors_unique:
#         predict[f] = 0
#     virtual_machine_sum = 0
#     #end modify

#     predict_days = (predict_end_time-predict_start_time).days
#     sample,mapping_index = resample(ecs_logs,flavors_unique,training_start_time,predict_start_time,frequency='{}d'.format(predict_days))

#     for f in flavors_unique:
#         predict[f] = int(round(sample[0][mapping_index[f]]))
#         virtual_machine_sum += int(round(sample[0][mapping_index[f]]))

#     return predict,virtual_machine_sum

def predict_flavors_unique_linear_regression(ecs_logs,flavors_unique,training_start_time,training_end_time,predict_start_time,predict_end_time):
    # modify @ 2018-03-15 
    predict = {}.fromkeys(flavors_unique)
    for f in flavors_unique:
        predict[f] = 0
    virtual_machine_sum = 0
    #end modify

    predict_days = (predict_end_time-predict_start_time).days
    sample,mapping_index = resample(ecs_logs,flavors_unique,training_start_time,predict_start_time,frequency='{}d'.format(predict_days))

    for f in flavors_unique:
        predict[f] = int(round(sample[0][mapping_index[f]]))
        virtual_machine_sum += int(round(sample[0][mapping_index[f]]))
        
    return predict,virtual_machine_sum


def predict_flavors_unique_correalation_method(ecs_logs,flavors_unique,training_start_time,training_end_time,predict_start_time,predict_end_time):

    # modify @ 2018-03-15 
    predict = {}.fromkeys(flavors_unique)
    for f in flavors_unique:
        predict[f] = 0
    virtual_machine_sum = 0
    #end modify

    predict_days = (predict_end_time-predict_start_time).days

    sample,mapping_index = resample(ecs_logs,flavors_unique,training_start_time,predict_start_time,frequency='{}d'.format(predict_days))
    prior_correlation_filter_0,prior_mapping_index = _corr_prior(flavors_unique)

    prior_correlation, prior_mapping_index = _corr_prior(flavors_unique)
    transform_sample_1 = matrix_mutmul(sample,prior_correlation)

    prior_correlation = correlation(sample)
    transform_sample_2 = matrix_mutmul(sample,prior_correlation)

    for i in range(len(prior_correlation)):
        for j in range(len(prior_correlation[0])):
            if prior_correlation[i][j]<-0.0:
                prior_correlation[i][j] = 0 

    assert(mapping_index==prior_mapping_index)
    merge_1 = [0.3*sample[0][i]+0.7*transform_sample_1[0][i] for i in range(len(sample[0]))]
    merge_2 = [0.9*sample[0][i]+0.1*transform_sample_2[0][i] for i in range(len(sample[0]))]

    merge = [0.2*merge_1[i] + 0.8*merge_2[i] for i in range(len(merge_1))]
    # merge = [0.4*merge_1[i] + 0.6*merge_2[i] for i in range(len(merge_1))]

    for f in flavors_unique:
        predict[f] = int(round(merge[mapping_index[f]]))
        virtual_machine_sum += int(round(merge[mapping_index[f]]))
    return predict,virtual_machine_sum


def predict_flavors_unique(ecs_logs,flavors_unique,training_start_time,training_end_time,predict_start_time,predict_end_time):
    # return predict_flavors_unique_correalation_method(ecs_logs,flavors_unique,training_start_time,training_end_time,predict_start_time,predict_end_time)
    return predict_flavors_unique_linear_regression(ecs_logs,flavors_unique,training_start_time,training_end_time,predict_start_time,predict_end_time)

    predict = {}.fromkeys(flavors_unique)
    for f in flavors_unique:
        predict[f] = 0
    virtual_machine_sum = 0
    
    if len(ecs_logs) == 0:
        return predict,virtual_machine_sum 

    predict_days = (predict_end_time-predict_start_time).days
    training_days = (training_end_time - training_start_time).days

    _double11 = datetime.strptime('{}-11-11 00:00:00'.format(predict_start_time.year), "%Y-%m-%d %H:%M:%S")
    _double12 = datetime.strptime('{}-12-12 00:00:00'.format(predict_start_time.year), "%Y-%m-%d %H:%M:%S")

    max_ma = training_days//predict_days
    if max_ma ==0:
        return predict,virtual_machine_sum 

    ma = 1
    assert(ma<=max_ma) #passed

    for i in flavors_unique:
        c = [True if flavor==i and (training_end_time-ecs_time).days<predict_days*ma else False for flavor,ecs_time in ecs_logs]
# --------------------------------------------testing below------------------------ #
        # # controling of c
        # count = 0
        # for j in range(len(c)):
        #     if c[j]:
        #         count+=1
        #     else:
        #         count=0
        #     if count>50:
        #         k = j
        #         while k<len(c) and c[k]:
        #             c[k] = False
        #             k+=1
# --------------------------------------------testing above------------------------ #

        if(predict_start_time<_double12 and predict_end_time>_double12 or predict_start_time<_double11 and predict_end_time>_double11):
            # print('special time')
            scale_time = 5
            predict[i] += int(round(count_nonezero(c)/ma)) * scale_time
            virtual_machine_sum += int(round(count_nonezero(c)/ma)) *scale_time
        else:
            predict[i] = int(round(count_nonezero(c)/ma)) 
            virtual_machine_sum += int(round(count_nonezero(c)/ma)) 

    # for i in flavors_unique:
    #     print(flavors[i],predict[i])

    return predict,virtual_machine_sum

def backpack(machine_config,flavors,flavors_unique,predict,is_random=False):
    entity_machine_sum = 0
    backpack_list = []

    current_machine_cpu = machine_config['cpu']
    current_machine_mem = machine_config['mem']

    m = {}
    m.fromkeys(flavors_unique)
    for i in flavors_unique:
        m[i] = 0

    em = dict(m)

    vm_list_normal = []
    for k,v in predict.items():
        vm_list_normal.extend([k for _ in range(v)])

    # shuffle virtual machine orders
    vm_list_random = [x for x in vm_list_normal]
    random.shuffle(vm_list_random)

    if is_random:
        vm_list = vm_list_random
    else:
        vm_list = vm_list_normal

    for i in vm_list:
        # try to push into backpack pool
        is_fit = False
        for p_em in backpack_list:
            cpu,mem = _get_em_weights_of_cpu_and_mem(p_em,flavors)
            if(machine_config['cpu']-cpu>=flavors[i][0] and machine_config['mem']-mem>=flavors[i][1]):
                p_em[i]+=1
                is_fit = True
                break
        if is_fit:
            continue

        if current_machine_cpu >= flavors[i][0] and current_machine_mem >= flavors[i][1]:
            em[i] += 1
            current_machine_cpu-=flavors[i][0]
            current_machine_mem-=flavors[i][1]
            # print(current_machine_cpu,current_machine_mem)
        else:
            backpack_list.append(em)
            entity_machine_sum+=1
            em = dict(m)
            current_machine_cpu = machine_config['cpu']
            current_machine_mem = machine_config['mem']
            em[i] += 1
            current_machine_cpu-=flavors[i][0]
            current_machine_mem-=flavors[i][1]

    if(count_nonezero(em.values())!=0):
        backpack_list.append(em)
        entity_machine_sum+=1
    # print(backpack_list)
    return backpack_list,entity_machine_sum

def backpack_random_k_times(machine_config,flavors,flavors_unique,predict,k=3):
    assert(k>=0)
    solution_set = []
    solution_set.append(backpack(machine_config,flavors,flavors_unique,predict))

    for i in range(k):
        solution_set.append(backpack(machine_config,flavors,flavors_unique,predict,is_random=True))

    best_solution = None
    minium_count = None

    # print([x[1] for x in solution_set])

    for s in solution_set:
        if not minium_count:
            minium_count = s[1]
            best_solution = s
        elif minium_count>s[1]:
            minium_count = s[1]
            best_solution = s
    return best_solution

def _get_em_weights_of_cpu_and_mem(em,flavors):
    cpu = 0
    mem = 0
    for k,v in em.items():
        cpu += flavors[k][0]*v
        mem += flavors[k][1]*v
    return cpu,mem

def _convert_machine_string(em):
    s = ""
    for k,v in em.items():
        if v != 0:
            s += " flavor{} {}".format(k,v)
    return s

def predict_vm(ecs_lines,input_lines):
    result = []

    if input_lines is None or ecs_lines is None:
        return []

    machine_config,flavors_number,flavors,flavors_unique,optimized,predict_start_time,predict_end_time = parse_input_lines(input_lines)
    ecs_logs,training_start_time,training_end_time = parse_ecs_lines(ecs_lines)


    predict,virtual_machine_sum = predict_flavors_unique(ecs_logs,flavors_unique,training_start_time,training_end_time,predict_start_time,predict_end_time)


    result.append('{}'.format(virtual_machine_sum))
    for i in flavors_unique:
        result.append('flavor{} {}'.format(i,predict[i]))

    result.append('') # output '\n'

    # backpack_list,entity_machine_sum = backpack(machine_config,flavors,flavors_unique,predict)
    backpack_list,entity_machine_sum = backpack_random_k_times(machine_config,flavors,flavors_unique,predict,k=10)
    result.append('{}'.format(entity_machine_sum))

    # print(backpack_list)

    c = 1
    for em in backpack_list:
        result.append(str(c)+_convert_machine_string(em))
        c += 1

    # print(result)
    return result



