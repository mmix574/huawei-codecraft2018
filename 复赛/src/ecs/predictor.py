import math
import random
import re
from datetime import datetime, timedelta

from learn.ridge import Ridge
from linalg.common import (abs, apply, dim, fancy, mean, minus, reshape, shape,
                           sqrt, sum, zeros)
from linalg.matrix import hstack, stdev
from linalg.vector import arange

# checked
def parse_input_lines(input_lines):
    # strip each line
    input_lines = [x.strip() for x in input_lines]
    
    machine_number = 0
    machine_name = []
    machine_config = []

    flavors_number = 0
    flavors_config = []

    predict_start = None
    predict_end = None 
    
    # 1,2,5,8 for example
    flavors_unique = []

    part = 1
    for line in input_lines:
        if line == '':
            part += 1
            continue

        if part == 1:
            machine_number = int(line)
            part +=1
        
        elif part == 2:
            # ['General', 'Large-Memory', 'High-Performance']
            machine_name.append(line.split(' ')[0])
            
            # [{'MEM': '128', 'CPU': '56'}, {'MEM': '256', 'CPU': '84'}, {'MEM': '192', 'CPU': '112'}]
            machine_config.append({'CPU':int(line.split(' ')[1]),'MEM':int(line.split(' ')[2])})

        elif part == 3:
            flavors_number = int(line)
            part+=1
        
        elif part==4:
            conf = re.findall('\d+',line)
            conf = [int(c) for c in conf]
            conf[2]/=1024

            flavors_unique.append(conf[0])
            flavors_config.append({'CPU':conf[1],'MEM':conf[2]})

        elif part == 5:
            predict_start = datetime.strptime(line, "%Y-%m-%d %H:%M:%S")
            part+=1

        elif part==6:
            predict_end = datetime.strptime(line, "%Y-%m-%d %H:%M:%S")


    # safty consideration
    flavors_unique = sorted(flavors_unique)


    # returns:
    # 
    # (3, ['General', 'Large-Memory', 'High-Performance'], 
    # [{'MEM': 0, 'CPU': 56}, {'MEM': 0, 'CPU': 84}, {'MEM': 0, 'CPU': 112}], 
    # 5, 
    # [1, 2, 4, 5, 8], 
    # [{'MEM': 1, 'CPU': 1}, {'MEM': 2, 'CPU': 1}, {'MEM': 2, 'CPU': 2}, {'MEM': 4, 'CPU': 2}, {'MEM': 8, 'CPU': 4}], 
    # datetime.datetime(2016, 1, 8, 0, 0), 
    # datetime.datetime(2016, 1, 14, 23, 59, 59))

    return machine_number,machine_name,machine_config,flavors_number,flavors_unique,flavors_config,predict_start,predict_end


# checked
def parse_ecs_lines(ecs_lines,flavors_unique):
    ecs_lines = [l.strip() for l in ecs_lines]
    ecs_logs = []
    if(len(ecs_lines)==0):
        return ecs_logs,None,None
        
    for line in ecs_lines:
        _uuid,f,t = line.split('\t')
        f = int(f[f.find('r')+1:])
        
        # add 2018-04-09 
        if f not in flavors_unique:
            continue
        
        t = datetime.strptime(t.strip(), "%Y-%m-%d %H:%M:%S")
        ecs_logs.append((f,t))

    training_start = ecs_logs[0][1]
    training_end = ecs_logs[len(ecs_logs)-1][1]

    # returns:
    # 
    #[(1, datetime.datetime(2015, 12, 31, 17, 54, 32)), (3, datetime.datetime(2015, 12, 31, 17, 54, 39)), (8, datetime.datetime(2015, 12, 31, 17, 54, 48)), (8, datetime.datetime(2015, 12, 31, 17, 54, 54)), (8, datetime.datetime(2015, 12, 31, 20, 9, 50)), (4, datetime.datetime(2015, 12, 31, 22, 13, 45))]
    # datetime.datetime(2015, 12, 1, 1, 14, 34),
    # datetime.datetime(2015, 12, 31, 22, 13, 45)

    return ecs_logs,training_start,training_end


# add @2018-04-10
# refactoring, do one thing.
def resampling(ecs_logs,flavors_unique,training_start_time,predict_start_time,frequency=7,strike=1,skip=0):
    # checked
    def __get_flavors_unique_mapping(flavors_unique):
        mapping_index = {}.fromkeys(flavors_unique)
        c = 0
        for f in flavors_unique:
            mapping_index[f] = c
            c+=1
        return mapping_index

    predict_start_time = predict_start_time-timedelta(days=skip)
    days_total = (predict_start_time-training_start_time).days

    sample_length = ((days_total-frequency)/strike) + 1
    mapping_index = __get_flavors_unique_mapping(flavors_unique)

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


def predict_flavors(ecs_logs,flavors_config,flavors_unique,training_start,training_end,predict_start,predict_end):
    predict_days = (predict_end-predict_start).days #check
    hours = ((predict_end-predict_start).seconds/float(3600))
    if hours >= 12:
        predict_days += 1
    
    sample = resampling(ecs_logs,flavors_unique,training_start,training_end,frequency=predict_days,strike=predict_days,skip=0)
    # problem #1 here
    def outlier_handling(sample,method='mean',max_sigma=3):
        assert(method=='mean')
        std_ = stdev(sample)
        mean_ = mean(sample,axis=1)
        for i in range(shape(sample)[0]):
            for j in range(shape(sample)[1]):
               if sample[i][j]-mean_[j] >max_sigma*std_[j]:
                    if method=='mean':
                        sample[i][j] = mean_[j]
        return sample

    # sample = outlier_handling(sample,method='mean',max_sigma=3)
    # from preprocessing import exponential_smoothing
    # sample = exponential_smoothing(sample)
    # sample = sample[-7:]

    prediction = []
    skip_days = (predict_start-training_end).days
    for i in range(shape(sample)[1]):
        clf = Ridge(alpha=1)
        X = reshape(list(range(len(sample))),(-1,1))
        y = fancy(sample,None,(i,i+1))

        # unbias estimation
        X_test = [[len(sample)+skip_days]]
        # X = hstack([X,apply(X,lambda x:x**2),apply(X,lambda x:math.pow(x,3))])
        # X_test = hstack([X_test,apply(X_test,lambda x:x**2),apply(X_test,lambda x:math.pow(x,3))])
        
        X = hstack([X,apply(X,lambda x:math.log1p(x)),sqrt(X)])
        X_test = hstack([X_test,apply(X_test,lambda x:math.log1p(x)),sqrt(X_test)])
        clf.fit(X,y)
        p = clf.predict(X_test)
        prediction.extend(p[0])
    
    prediction = [int(round(p)) if p>0 else 0 for p in prediction]
    return prediction


# add @2018-04-16
def argmin(A):
    assert(dim(A)==1)
    min_index = 0
    for i in range(len(A)):
        if A[i] < A[min_index]:
            min_index = i
    return min_index



def backpack(machine_number,machine_name,machine_config,flavors_number,flavors_unique,flavors_config,prediction,is_random=False):
    # parameters:
    # machine_number,machine_name,machine_config,flavors_number,flavors_unique,flavors_config,prediction
    # -->
    # (3, 
    # ['General', 'Large-Memory', 'High-Performance'], 
    # [{'MEM': 128, 'CPU': 56}, {'MEM': 256, 'CPU': 84}, {'MEM': 192, 'CPU': 112}],
    #  5, 
    # [1, 2, 4, 5, 8], 
    # [{'MEM': 1, 'CPU': 1}, {'MEM': 2, 'CPU': 1}, {'MEM': 2, 'CPU': 2}, {'MEM': 4, 'CPU': 2}, {'MEM': 8, 'CPU': 4}], 
    # [32, 32, 11, 21,44])
    

    machine_rate = [c['CPU']/float(c['MEM']) for c in machine_config]

    cpu_predict = 0
    mem_prefict = 0
    for i in range(len(prediction)):
        cpu_predict += (prediction[i] * flavors_config[i]['CPU'])
        mem_prefict += (prediction[i] * flavors_config[i]['MEM'])

    type_i = argmin(abs(minus(machine_rate,cpu_predict/float(mem_prefict))))
    
    vms = []
    for i in range(len(prediction)):
        f_config = flavors_config[i] 
        vms.extend(
            [
                [flavors_unique[i],{'CPU':f_config['CPU'],'MEM':f_config['MEM']}] 
            for _ in range(prediction[i])]
            )

    if is_random:
        from random import shuffle
        shuffle(vms)
    # vms:
    # [(1, {'MEM': 1, 'CPU': 1}), (2, {'MEM': 1, 'CPU': 1}), (4, {'MEM': 1,'CPU': 1}), (5, {'MEM': 1, 'CPU': 1}), (8, {'MEM': 1, 'CPU': 1}), (1,{'MEM': 2, 'CPU': 1}), (2, {'MEM': 2, 'CPU': 1}), (4, {'MEM': 2, 'CPU': 1}), (5, {'MEM': 2, 'CPU': 1}), (8, {'MEM': 2, 'CPU': 1}), (1, {'MEM': 2, 'CPU': 2}), (2, {'MEM': 2, 'CPU': 2}), (4, {'MEM': 2, 'CPU': 2}), (5, {'MEM': 2, 'CPU': 2}), (8, {'MEM': 2, 'CPU': 2}), (1, {'MEM': 4, 'CPU': 2}), (2, {'MEM': 4, 'CPU': 2}), (4, {'MEM': 4, 'CPU': 2}), (5, {'MEM': 4, 'CPU': 2}), (8, {'MEM': 4, 'CPU': 2}), (1, {'MEM': 8, 'CPU': 4}), (2, {'MEM': 8, 'CPU': 4}), (4, {'MEM': 8, 'CPU': 4}), (5, {'MEM': 8, 'CPU': 4}), (8, {'MEM': 8, 'CPU': 4})]


    # [[{f1:3,f5:2},{f8:2,f7:4}] <== type1 machine
    # [....]                     <== type2 machine
    # [....]                     <== type3 machine
    backpack_result = [[] for _ in range(machine_number)]

    placing = [None for _ in range(machine_number)]

    def _get_em_weights_of_cpu_and_mem(flavors_unique,flavors_config,em):
        cpu = 0
        mem = 0
        for k,v in em.items():
            cpu += flavors_config[flavors_unique.index(k)]['CPU']*v
            mem += flavors_config[flavors_unique.index(k)]['MEM']*v
        return cpu,mem
    
    while(len(vms)!=0):
        vm_flavor = vms[0][0]
        vm_config = vms[0][1]
        if placing[type_i] == None:
            placing[type_i] = {}.fromkeys(flavors_unique)
            for f in flavors_unique:
                placing[type_i][f] = 0 
            continue
        else:
            cpu_total,mem_total = machine_config[type_i]['CPU'],machine_config[type_i]['MEM']
            cpu_used,mem_used = _get_em_weights_of_cpu_and_mem(flavors_unique,flavors_config,placing[type_i])
            if cpu_total-cpu_used<vm_config['CPU'] or mem_total-mem_used<vm_config['MEM']:
                backpack_result[type_i].append(placing[type_i])
                placing[type_i] = None
            else:
                placing[type_i][vm_flavor]+=1
                vms.pop(0)
                
                # add @2018-04-18
                # select next type of entity machine
                type_i = random.choice(range(machine_number))

    for i in range(len(placing)):
        if placing[i]!=None:

            # add @2018-04-18
            cpu_used,mem_used = _get_em_weights_of_cpu_and_mem(flavors_unique,flavors_config,placing[type_i])
            if cpu_used!=0 and mem_used!=0:
                backpack_result[i].append(placing[i])

    backpack_count = [len(b) for b in backpack_result]

    return backpack_count,backpack_result



# build output lines
def predict_vm(ecs_lines,input_lines):
    if input_lines is None or ecs_lines is None:
        return []

    machine_number,machine_name,machine_config,flavors_number,flavors_unique,flavors_config,predict_start,predict_end = parse_input_lines(input_lines)
    ecs_logs,training_start,training_end = parse_ecs_lines(ecs_lines,flavors_unique)

    prediction = predict_flavors(ecs_logs,flavors_config,flavors_unique,training_start,training_end,predict_start,predict_end)
    # flavors_unique:
    # [1, 2, 4, 5, 8]

    # prediction:
    # [32, 32, 11, 21, 44]

    # flavors_config:
    # [{'MEM': 1, 'CPU': 1}, {'MEM': 2, 'CPU': 1}, {'MEM': 2,'CPU': 2}, {'MEM': 4, 'CPU': 2}, {'MEM': 8, 'CPU': 4}]
    # backpack_count,backpack_result = backpack(machine_number,machine_name,machine_config,flavors_number,flavors_unique,flavors_config,prediction)
    
    print(machine_config)


    min_count = None
    best_result = None

    def get_backpack_score(machine_number,machine_config,flavors_unique,flavors_config,backpack_result):
        def _get_em_weights_of_cpu_and_mem(flavors_unique,flavors_config,em):
            cpu = 0
            mem = 0
            for k,v in em.items():
                cpu += flavors_config[flavors_unique.index(k)]['CPU']*v
                mem += flavors_config[flavors_unique.index(k)]['MEM']*v
            return cpu,mem
        
        cpu_total_total = 0
        mem_total_total = 0
        cpu_used_total_total = 0
        mem_used_total_total = 0
        for i in range(machine_number):
            cpu_total = len(backpack_result[i])*machine_config[i]['CPU']
            mem_total = len(backpack_result[i])*machine_config[i]['MEM']
            cpu_total_total += cpu_total
            mem_total_total += cpu_total

            # state:[(cpu,mem),(cpu,mem)...]
            # [(81, 155), (82, 159), (84, 157), (81, 153)]
            print(backpack_result[i])
            state = [_get_em_weights_of_cpu_and_mem(flavors_unique,flavors_config,em) for em in backpack_result[i]]
            print(state)
            cpu_used_total = sum([s[0] for s in state])
            mem_used_total = sum([s[1] for s in state])

            cpu_used_total_total += cpu_used_total
            mem_used_total_total += mem_used_total


            print(cpu_used_total,cpu_total_total)
            # print(mem_used_total,mem_total_total)

        cpu_rate = cpu_used_total_total/float(cpu_total_total)
        mem_rate = mem_used_total_total/float(mem_total_total)
        return cpu_rate,mem_rate

    for i in range(1000):
        backpack_count,backpack_result = backpack(machine_number,machine_name,machine_config,flavors_number,flavors_unique,flavors_config,prediction,is_random=True)
        cpu_rate,mem_rate = get_backpack_score(machine_number,machine_config,flavors_unique,flavors_config,backpack_result)
        # print(cpu_rate,mem_rate)
        if not min_count or min_count>backpack_count:
            min_count = backpack_count
            best_result = backpack_result
        exit()
    backpack_count = min_count
    backpack_result = best_result

    result = []
    result.append('{}'.format(sum(prediction)))
    for i in range(len(prediction)):
        result.append('flavor{} {}'.format(flavors_unique[i],prediction[i]))

    # General  2
    # General-1  flavor5  2
    # General-2    1  flavor10  1
    # Large-Memory  1
    # Large-Memory-1  flavor10  1
    # High-Performance  1
    # High-Performance-1  flavor15  1

    def _convert_machine_string(em):
        s = ""
        for k,v in em.items():
            if v != 0:
                s += " flavor{} {}".format(k,v)
        return s

    for i in range(machine_number):
        c = 1
        if backpack_count[i]!=0:

            result.append('') # output '\n'
            
            result.append('{} {}'.format(machine_name[i],backpack_count[i]))
            for em in backpack_result[i]:
                result.append('{}-{}{}'.format(machine_name[i],c,_convert_machine_string(em)))
                c += 1

    return result
