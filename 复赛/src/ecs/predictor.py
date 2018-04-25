import math
import random
import re
from datetime import datetime, timedelta

from learn.ridge import Ridge
from linalg.common import (abs, apply, dim, fancy, flatten, mean, minus,
                           reshape, shape, sqrt, square, sum, zeros)
from linalg.matrix import hstack, stdev
from linalg.vector import arange, argmax, argmin

# change lucky random seed.
random.seed(3)

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
    
    last_time = [None for i in range(len(flavors_unique))]

    for i in range(sample_length):
        for f,ecs_time in ecs_logs:
            # 0 - 6 for example
            # fix serious bug @ 2018-04-11
            if (predict_start_time-ecs_time).days >=(i)*strike and (predict_start_time-ecs_time).days<(i)*strike+frequency:
                if last_time[mapping_index[f]] == None:
                    sample[i][mapping_index[f]] += 1
                    last_time[mapping_index[f]] = ecs_time

                else:
                    if (ecs_time-last_time[mapping_index[f]]).seconds<10:
                        sample[i][mapping_index[f]] += 1
                        continue
                    else:
                        sample[i][mapping_index[f]] += 1
                        last_time[mapping_index[f]] = ecs_time

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
    
    skip_days = (predict_start-training_end).days

    # print(skip_days) #checked
    # print(predict_days) #checked

    # sample = resampling(ecs_logs,flavors_unique,training_start,training_end,frequency=predict_days,strike=predict_days,skip=0)
    sample = resampling(ecs_logs,flavors_unique,training_start,training_end,frequency=1,strike=1,skip=0)

    def outlier_handling(sample,method='mean',max_sigma=3):
        assert(method=='mean' or method=='dynamic')
        std_ = stdev(sample)
        mean_ = mean(sample,axis=0)
        for i in range(shape(sample)[0]):
            for j in range(shape(sample)[1]):
               if sample[i][j]-mean_[j] >max_sigma*std_[j]:
                    if method=='mean':
                        sample[i][j] = mean_[j]
                    elif method=='dynamic':
                        if i<len(sample)/2.0:
                            sample[i][j] = (mean_[j] + sample[i][j])/2.0
        return sample

    # sample = outlier_handling(sample,method='dynamic',max_sigma=3)
    # sample = outlier_handling(sample,method='mean',max_sigma=3.5)
    
    # from preprocessing import exponential_smoothing
    # sample = exponential_smoothing(exponential_smoothing(sample,alpha=0.2),alpha=0.2)
    prediction = []
    for i in range(shape(sample)[1]):

        clf = Ridge(alpha=1,fit_intercept=False)

        X = reshape(list(range(len(sample))),(-1,1))
        y = fancy(sample,None,(i,i+1))

        X_test = reshape(list(range(len(sample)+skip_days,len(sample)+skip_days+predict_days)),(-1,1))

        X_list = [X]
        X = hstack(X_list)
        
        X_test_list = [X_test]
        X_test = hstack(X_test_list)

        clf.fit(X,y)
        p = clf.predict(X_test)


        prediction.append(sum(flatten(p)))

    prediction = [int(round(p)) if p>0 else 0 for p in prediction]

    return prediction




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
    mem_predict = 0
    for i in range(len(prediction)):
        cpu_predict += (prediction[i] * flavors_config[i]['CPU'])
        mem_predict += (prediction[i] * flavors_config[i]['MEM'])

    type_i_fix = argmin(abs(minus(machine_rate,cpu_predict/float(mem_predict))))
    
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

    # same size of backpack_result,for reduce repected calclation
    backpack_capcity = [[] for _ in range(machine_number)]


    placing = [None for _ in range(machine_number)]

    def _get_em_weights_of_cpu_and_mem(flavors_unique,flavors_config,em):
        cpu = 0
        mem = 0
        for k,v in em.items():
            cpu += flavors_config[flavors_unique.index(k)]['CPU']*v
            mem += flavors_config[flavors_unique.index(k)]['MEM']*v
        return cpu,mem


    type_i = type_i_fix
    while(len(vms)!=0):
        vm_flavor = vms[0][0]
        vm_config = vms[0][1]
        # ------------------refiting ------------------------------  
        refit = False
        insert_order = list(range(machine_number))
        # shuffle(insert_order)
        for i in insert_order:
            for j in range(len(backpack_result[i])):
                cpu_cap,mem_cap = backpack_capcity[i][j]
                if cpu_cap>=vm_config['CPU'] and mem_cap>=vm_config['MEM']:
                    backpack_result[i][j][vm_flavor]+=1

                    # used for estimate the cpu/mem rate
                    cpu_predict -= vm_config['CPU']
                    mem_predict -= vm_config['MEM']

                    # success
                    backpack_capcity[i][j] = cpu_cap-vm_config['CPU'],mem_cap-vm_config['MEM']
                    refit = True
                    break
            if refit:
                break
        if refit:
            vms.pop(0)
            continue
        # -------------------normal fitting------------------------
        if placing[type_i] == None:
            placing[type_i] = {}.fromkeys(flavors_unique)
            for f in flavors_unique:
                placing[type_i][f] = 0 
            continue
        else:
            cpu_total,mem_total = machine_config[type_i]['CPU'],machine_config[type_i]['MEM']
            cpu_used,mem_used = _get_em_weights_of_cpu_and_mem(flavors_unique,flavors_config,placing[type_i])
            if cpu_total-cpu_used<vm_config['CPU'] or mem_total-mem_used<vm_config['MEM']:
                # add to backpack_list and create a new entity_machine
                backpack_result[type_i].append(placing[type_i])
                backpack_capcity[type_i].append((cpu_total-cpu_used,mem_total-mem_used))

                placing[type_i] = None
            else:
                placing[type_i][vm_flavor]+=1
                
                # used for estimate the cpu/mem rate
                cpu_predict -= vm_config['CPU']
                mem_predict -= vm_config['MEM']

                vms.pop(0)
                
                # add @2018-04-18
                # select next type of entity machine
                # type_i = random.choice(range(machine_number))

                if mem_predict==0:
                    break
                # 1.Greedy Select
                type_i = argmin(abs(minus(machine_rate,cpu_predict/float(mem_predict))))


    for i in range(len(placing)):
        if placing[i]!=None:

            # add @2018-04-18
            cpu_used,mem_used = _get_em_weights_of_cpu_and_mem(flavors_unique,flavors_config,placing[i])
            if cpu_used!=0 and mem_used!=0:
                possible = []
                for k in range(machine_number):
                    if machine_config[k]['CPU']>=cpu_used and machine_config[k]['MEM']>=mem_used:
                        possible.append(True)
                    else:
                        possible.append(False)
                scores = [(cpu_used/float(machine_config[k]['CPU']) + mem_used/float(machine_config[k]['MEM']))/2.0 if possible[k] else 0 for k in range(machine_number)]

                best_i = argmax(scores)
                backpack_result[best_i].append(placing[i])
                
                # backpack_result[i].append(placing[i])

    backpack_count = [len(b) for b in backpack_result]

    return backpack_count,backpack_result


def get_approximate_meta_solutions(machine_number,machine_name,machine_config,flavors_number,flavors_unique,flavors_config,prediction,max_iter=1000,score_treadhold=0.99):
    meta_solu = []

    vms = []
    for i in range(len(prediction)):
        f_config = flavors_config[i] 
        vms.extend(
            [
                [flavors_unique[i],{'CPU':f_config['CPU'],'MEM':f_config['MEM']}] 
            for _ in range(prediction[i])]
            )
    
    def get_solu_cpu_mem(solu,flavors_unique,flavors_config):
        cpu = 0 
        mem = 0 
        for i in range(len(flavors_unique)):
            cpu += solu[i]*flavors_config[i]['CPU']
            mem += solu[i]*flavors_config[i]['MEM']
        return cpu,mem

    def estimate_partial_score(config,solu,flavors_unique,flavors_config):
        cpu_total = config['CPU']
        mem_total = config['MEM']
        
        cpu_used,mem_used = get_solu_cpu_mem(solu,flavors_unique,flavors_config)
        # print(cpu_used,mem_used)
        cpu_rate = cpu_used/float(cpu_total)
        mem_rate = mem_used/float(mem_total)
        return (cpu_rate+mem_rate)/2.0


    def generate_single_prediction_based(config,flavors_unique,flavors_config,vms,max_iter=1000,score_treadhold=0.99):
        result = set()
        for i in range(max_iter):
            cpu = config['CPU']
            mem = config['MEM']
            from random import shuffle
            shuffle(vms)
            
            # [f1,f3,f5,f8,f9] <--flavor unique
            #  |  |  |  |  |
            # solu:
            # [0,1,2,3,4,5]<-- index
            # (3,4,7,1,0) <-- count
            solu = [ 0 for _ in range(len(flavors_unique))]  

            for vm in vms:
                if cpu>=vm[1]['CPU'] and mem>=vm[1]['MEM']:
                    solu[flavors_unique.index(vm[0])] += 1
                    cpu-=vm[1]['CPU'] 
                    mem-=vm[1]['MEM']
            
            solu = tuple(solu) #hashable tuple

            score = estimate_partial_score(config,solu,flavors_unique,flavors_config)
            
            if score >= score_treadhold:
                result.add(solu)
        return result

    def generate_single_expert_based(config,flavors_unique,flavors_config,vms,max_iter=1000,score_treadhold=0.99):
        result = set()

        for _ in range(max_iter):
            cpu = config['CPU']
            mem = config['MEM']
            
            solu = [ 0 for _ in range(len(flavors_unique))] 
            def still_can_fit(cpu,mem,flavors_config):
                for i in range(len(flavors_config)):
                    if flavors_config[i]['CPU']<=cpu and flavors_config[i]['MEM']<=mem:
                        return True              
                return False

            full = False
            while not full:
                f_i = random.choice(range(len(flavors_unique)))
                if flavors_config[f_i]['CPU']<=cpu and flavors_config[f_i]['MEM']<=mem:
                    cpu-=flavors_config[f_i]['CPU']
                    mem-=flavors_config[f_i]['MEM']
                    solu[f_i]+=1
                full = not still_can_fit(cpu,mem,flavors_config)
            solu = tuple(solu)
            score = estimate_partial_score(config,solu,flavors_unique,flavors_config)
            if score>score_treadhold and len(solu)!=0:
                result.add(solu)
        
        return result

    for i in range(machine_number):
        solu = generate_single_prediction_based(machine_config[i],flavors_unique,flavors_config,vms,max_iter=max_iter,score_treadhold=score_treadhold)
        # solu = generate_single_expert_based(machine_config[i],flavors_unique,flavors_config,vms,max_iter=1000,score_treadhold=score_treadhold)
        
        meta_solu.append(solu)

    return meta_solu


# warpper function for 
def random_k_times(machine_number,machine_name,machine_config,flavors_number,flavors_unique,flavors_config,prediction,k=50):
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
            mem_total_total += mem_total

            # state:[(cpu,mem),(cpu,mem)...]
            # [(81, 155), (82, 159), (84, 157), (81, 153)]
            state = [_get_em_weights_of_cpu_and_mem(flavors_unique,flavors_config,em) for em in backpack_result[i]]
            cpu_used_total = sum([s[0] for s in state])
            mem_used_total = sum([s[1] for s in state])

            cpu_used_total_total += cpu_used_total
            mem_used_total_total += mem_used_total

            # print(cpu_used_total,cpu_total_total)
            # print(mem_used_total,mem_total_total)

        cpu_rate = cpu_used_total_total/float(cpu_total_total)
        mem_rate = mem_used_total_total/float(mem_total_total)
        return cpu_rate,mem_rate
    # end get_backpack_score function


    # maximize score
    max_score = None
    best_result = None
    min_count = None
    for i in range(k):
        backpack_count,backpack_result = backpack(machine_number,machine_name,machine_config,flavors_number,flavors_unique,flavors_config,prediction,is_random=True)
        # backpack_count,backpack_result = greedy_99_backpack(machine_number,machine_name,machine_config,flavors_number,flavors_unique,flavors_config,prediction)
        cpu_rate,mem_rate = get_backpack_score(machine_number,machine_config,flavors_unique,flavors_config,backpack_result)
        # find the best score solution 
        score  = (cpu_rate+mem_rate)/2.0
        # print(i,score)
        if not max_score or max_score<score:
            max_score = score
            best_result = backpack_result
            min_count = backpack_count
    # print("max_score-->",max_score)

    return min_count,best_result


def greedy_99_backpack(machine_number,machine_name,machine_config,flavors_number,flavors_unique,flavors_config,prediction,score_treadhold=0.99):
    backpack_result = None
    solutions = get_approximate_meta_solutions(machine_number,machine_name,machine_config,flavors_number,flavors_unique,flavors_config,prediction,score_treadhold=score_treadhold)

    # print(prediction)
    # print(solutions)
    
    def possible(prediction,picker):
        for i in range(len(prediction)):
            if picker[i]>prediction[i]:
                return False
        return True

    backpack_result = [[] for _ in range(machine_number)]

    fit = True
    while(fit):
        fit = False
        for i in range(len(solutions))[::-1]:
            pickers = solutions[i]
            for picker in pickers:
                picker = list(picker)
                if possible(prediction,picker):
                    prediction = minus(prediction,picker)
                    em = {}.fromkeys(flavors_unique)
                    for j in range(len(flavors_unique)):
                        em[flavors_unique[j]] = picker[j]
                    backpack_result[i].append(em)
                    fit = True

    # _,backpack_result_2 = backpack(machine_number,machine_name,machine_config,flavors_number,flavors_unique,flavors_config,prediction,is_random=True)
    _,backpack_result_2 = random_k_times(machine_number,machine_name,machine_config,flavors_number,flavors_unique,flavors_config,prediction,k=1000)


    # backpack merge
    for i in range(len(backpack_result)):
        backpack_result[i].extend(backpack_result_2[i])
        pass

    backpack_count = [len(b) for b in backpack_result]

    # backpack_count: entity machine sum
    # backpack_result:
    # [[{f1:3,f2:8}..etc....] 
    # [.......]
    # [.......]]
    return backpack_count,backpack_result



def greedy_general_backpack(machine_number,machine_name,machine_config,flavors_number,flavors_unique,flavors_config,prediction,is_random=True):
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

    while len(vms!=0):
        for config in machine_config:
            
            pass
        pass

    print(vms)
    print(len(vms))
    vms.remove([15, {'MEM': 64, 'CPU': 16}])
    print(len(vms))
    exit()

    def _get_em_weights_of_cpu_and_mem(flavors_unique,flavors_config,em):
        cpu = 0
        mem = 0
        for k,v in em.items():
            cpu += flavors_config[flavors_unique.index(k)]['CPU']*v
            mem += flavors_config[flavors_unique.index(k)]['MEM']*v
        return cpu,mem



    backpack_count = [len(b) for b in backpack_result]

    return backpack_count,backpack_result



# ----------------------------------backpack score--------------------------------------

def single_backpack_score(machine_config_single,flavors_unique,flavors_config,em):
    def _get_em_weights_of_cpu_and_mem(flavors_unique,flavors_config,em):
        cpu = 0
        mem = 0
        for k,v in em.items():
            cpu += flavors_config[flavors_unique.index(k)]['CPU']*v
            mem += flavors_config[flavors_unique.index(k)]['MEM']*v
        return cpu,mem
    
    cpu,mem = _get_em_weights_of_cpu_and_mem(flavors_unique,flavors_config,em)

    cpu_rate = cpu/float(machine_config_single['CPU'])
    mem_rate = mem/float(machine_config_single['MEM'])

    return (cpu_rate+mem_rate)/2.0

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
        mem_total_total += mem_total

        # state:[(cpu,mem),(cpu,mem)...]
        # [(81, 155), (82, 159), (84, 157), (81, 153)]
        state = [_get_em_weights_of_cpu_and_mem(flavors_unique,flavors_config,em) for em in backpack_result[i]]
        cpu_used_total = sum([s[0] for s in state])
        mem_used_total = sum([s[1] for s in state])

        cpu_used_total_total += cpu_used_total
        mem_used_total_total += mem_used_total

        # print(cpu_used_total,cpu_total_total)
        # print(mem_used_total,mem_total_total)

    cpu_rate = cpu_used_total_total/float(cpu_total_total)
    mem_rate = mem_used_total_total/float(mem_total_total)
    return cpu_rate,mem_rate

# build output lines
def predict_vm(ecs_lines,input_lines):
    if input_lines is None or ecs_lines is None:
        return []

    machine_number,machine_name,machine_config,flavors_number,flavors_unique,flavors_config,predict_start,predict_end = parse_input_lines(input_lines)
    ecs_logs,training_start,training_end = parse_ecs_lines(ecs_lines,flavors_unique)
    prediction = predict_flavors(ecs_logs,flavors_config,flavors_unique,training_start,training_end,predict_start,predict_end)
    
    # testing 
    # prediction  = [0, 24, 0, 0, 0, 210, 0, 0, 0, 0, 0, 18, 93, 45, 18, 0, 216, 0]
    # dp(machine_number,machine_name,machine_config,flavors_number,flavors_unique,flavors_config,prediction,
    # get_approximate_meta_solutions(machine_number,machine_name,machine_config,flavors_number,flavors_unique,flavors_config,prediction,score_treadhold=0.99,max_iter=100)
    # )
    # exit()
    
    predictions = [[0, 0, 0, 255, 0, 213, 0, 0, 248, 0, 0, 232, 76, 281, 210, 0, 235, 93],[0, 181, 0, 142, 0, 0, 99, 0, 0, 0, 48, 53, 198, 0, 0, 216, 130, 0],[0, 209, 105, 236, 0, 33, 0, 128, 11, 279, 73, 0, 0, 0, 202, 0, 0, 0],[0, 240, 0, 0, 182, 0, 208, 0, 10, 80, 52, 259, 89, 0, 0, 0, 171, 0],[0, 0, 0, 0, 0, 0, 0, 115, 179, 144, 0, 0, 0, 0, 26, 194, 0, 23],[0, 6, 0, 0, 0, 3, 0, 0, 0, 0, 0, 290, 173, 0, 0, 0, 258, 0],[96, 109, 0, 11, 0, 0, 169, 149, 0, 0, 0, 240, 86, 0, 0, 0, 0, 0],[0, 239, 0, 100, 298, 0, 0, 67, 140, 0, 23, 0, 240, 6, 139, 25, 0, 0],[163, 0, 0, 250, 0, 137, 14, 0, 81, 71, 0, 0, 50, 0, 0, 261, 0, 234],[0, 0, 0, 105, 0, 0, 0, 270, 0, 152, 264, 0, 120, 0, 50, 0, 270, 188],[0, 0, 37, 0, 0, 103, 0, 40, 190, 20, 207, 106, 0, 0, 178, 0, 89, 0],[0, 0, 0, 172, 164, 0, 0, 0, 9, 29, 95, 0, 0, 23, 284, 126, 0, 0],[239, 193, 68, 0, 0, 0, 0, 250, 217, 0, 206, 102, 81, 45, 55, 0, 0, 0],[0, 0, 0, 9, 38, 114, 238, 35, 0, 188, 0, 0, 0, 0, 0, 0, 0, 0],[0, 100, 110, 295, 248, 0, 0, 0, 138, 235, 252, 145, 0, 0, 0, 252, 233, 0],[0, 235, 0, 10, 5, 0, 0, 278, 177, 210, 251, 132, 224, 202, 0, 0, 0, 0],[0, 270, 0, 0, 0, 94, 0, 269, 0, 0, 92, 0, 0, 0, 0, 0, 123, 0],[0, 36, 0, 0, 0, 0, 110, 283, 104, 0, 5, 26, 0, 81, 0, 118, 0, 0],[0, 190, 66, 290, 0, 0, 25, 80, 37, 276, 10, 179, 44, 0, 0, 0, 259, 0],[77, 0, 0, 0, 124, 168, 15, 0, 142, 0, 0, 209, 0, 0, 160, 60, 192, 0],[43, 0, 287, 23, 0, 0, 95, 0, 0, 0, 0, 20, 110, 0, 0, 0, 0, 0],[54, 40, 0, 54, 0, 290, 0, 27, 0, 197, 0, 29, 0, 6, 0, 300, 45, 88],[0, 0, 0, 0, 249, 67, 0, 0, 207, 0, 169, 77, 0, 0, 0, 0, 115, 290],[0, 24, 0, 0, 0, 210, 0, 0, 0, 0, 0, 18, 93, 45, 18, 0, 216, 0],[208, 116, 0, 0, 0, 174, 220, 30, 9, 236, 104, 233, 0, 0, 0, 0, 285, 172],[0, 0, 293, 0, 0, 65, 42, 0, 45, 0, 141, 0, 163, 0, 0, 292, 0, 122],[169, 0, 0, 0, 0, 295, 228, 200, 137, 0, 0, 0, 0, 53, 22, 208, 123, 80],[0, 0, 78, 183, 147, 0, 291, 0, 0, 257, 96, 11, 36, 224, 0, 58, 77, 93],[284, 268, 12, 0, 60, 0, 127, 0, 178, 0, 0, 0, 0, 0, 0, 0, 0, 81],[0, 40, 0, 277, 0, 0, 0, 0, 0, 0, 51, 89, 138, 0, 125, 162, 219, 0],[226, 0, 0, 0, 176, 0, 42, 0, 20, 50, 189, 192, 31, 0, 210, 0, 141, 0],[179, 147, 92, 26, 0, 158, 57, 0, 232, 254, 0, 0, 278, 0, 0, 0, 230, 0],[0, 0, 262, 248, 0, 0, 159, 77, 0, 148, 47, 223, 0, 0, 209, 0, 12, 253],[0, 0, 0, 0, 230, 58, 0, 172, 0, 0, 0, 0, 0, 24, 0, 99, 0, 0],[294, 0, 61, 222, 0, 274, 0, 135, 10, 0, 287, 0, 0, 188, 0, 58, 103, 52],[0, 0, 0, 0, 53, 257, 0, 161, 43, 173, 230, 268, 0, 0, 179, 297, 23, 153],[0, 0, 0, 229, 0, 0, 26, 24, 0, 290, 0, 300, 0, 2, 94, 76, 163, 221],[0, 289, 166, 191, 179, 217, 121, 284, 0, 0, 249, 0, 0, 144, 0, 144, 245, 101],[0, 111, 0, 0, 0, 11, 0, 46, 0, 0, 188, 29, 47, 0, 0, 0, 57, 157],[0, 0, 257, 160, 0, 0, 227, 297, 77, 0, 0, 0, 52, 250, 0, 0, 0, 92],[0, 0, 0, 0, 105, 0, 0, 190, 139, 0, 0, 253, 0, 168, 65, 0, 0, 0],[50, 0, 205, 0, 0, 223, 129, 0, 0, 0, 0, 135, 0, 246, 143, 82, 165, 290],[0, 0, 0, 228, 293, 212, 66, 267, 0, 79, 0, 296, 22, 114, 287, 0, 0, 18],[278, 232, 90, 0, 0, 72, 273, 135, 126, 0, 0, 0, 115, 183, 147, 43, 79, 0],[110, 70, 57, 210, 97, 0, 102, 0, 163, 3, 275, 253, 10, 0, 0, 36, 168, 0],[0, 0, 0, 156, 102, 0, 0, 0, 0, 295, 212, 83, 43, 0, 0, 225, 0, 0],[0, 0, 211, 0, 0, 0, 15, 32, 0, 0, 224, 0, 128, 217, 0, 0, 105, 277],[0, 0, 0, 0, 11, 5, 0, 171, 292, 140, 0, 0, 181, 0, 0, 0, 0, 0],[0, 0, 0, 52, 0, 0, 88, 0, 57, 0, 199, 93, 118, 283, 0, 203, 62, 22],[0, 55, 0, 0, 0, 0, 0, 0, 29, 0, 0, 0, 0, 183, 0, 25, 0, 0],[0, 0, 258, 167, 0, 266, 0, 156, 0, 0, 168, 0, 52, 91, 123, 0, 0, 120],[0, 249, 0, 118, 6, 7, 17, 64, 295, 0, 123, 0, 0, 0, 296, 236, 0, 207],[87, 0, 276, 0, 175, 0, 225, 0, 0, 0, 191, 207, 154, 0, 0, 297, 182, 292],[58, 290, 65, 202, 63, 0, 0, 0, 135, 0, 39, 0, 42, 0, 0, 0, 39, 20],[0, 0, 79, 0, 0, 207, 283, 0, 137, 0, 0, 78, 269, 0, 165, 163, 290, 0],[0, 290, 115, 210, 0, 0, 45, 263, 0, 44, 0, 0, 0, 286, 186, 0, 0, 190],[88, 212, 0, 0, 176, 209, 0, 0, 14, 0, 45, 131, 0, 0, 0, 0, 0, 11],[257, 25, 0, 0, 0, 43, 0, 106, 186, 0, 265, 30, 0, 0, 0, 0, 113, 0],[295, 0, 0, 0, 0, 0, 0, 142, 212, 95, 49, 0, 0, 0, 0, 157, 0, 224],[0, 251, 0, 0, 181, 0, 0, 199, 85, 0, 0, 0, 0, 0, 185, 105, 192, 97],[269, 146, 0, 0, 0, 0, 0, 7, 0, 250, 0, 0, 71, 0, 0, 168, 257, 265],[148, 0, 227, 0, 124, 0, 155, 0, 0, 0, 221, 227, 0, 155, 0, 246, 241, 47],[0, 135, 169, 0, 27, 271, 8, 0, 45, 120, 0, 42, 0, 0, 0, 65, 0, 0],[38, 0, 0, 242, 127, 23, 0, 0, 200, 214, 0, 88, 277, 50, 0, 166, 100, 257],[61, 288, 253, 5, 0, 1, 0, 0, 133, 46, 0, 199, 0, 0, 284, 126, 285, 203],[134, 0, 27, 152, 0, 0, 0, 0, 162, 137, 0, 0, 241, 0, 272, 203, 12, 0],[0, 0, 0, 0, 0, 153, 0, 0, 0, 0, 0, 239, 0, 0, 204, 264, 0, 38],[0, 102, 98, 194, 0, 108, 0, 273, 0, 0, 74, 0, 0, 0, 112, 0, 0, 0],[0, 0, 102, 0, 200, 179, 0, 54, 0, 0, 0, 44, 182, 300, 0, 0, 196, 134],[0, 0, 0, 0, 0, 0, 285, 0, 157, 0, 162, 227, 0, 0, 203, 0, 0, 66],[118, 33, 200, 65, 12, 0, 0, 300, 165, 0, 219, 0, 209, 0, 250, 0, 0, 0],[0, 178, 0, 40, 34, 71, 60, 235, 0, 0, 43, 0, 120, 299, 55, 0, 0, 0],[0, 131, 0, 0, 287, 0, 287, 0, 127, 222, 0, 23, 0, 0, 63, 157, 180, 0],[0, 290, 266, 112, 186, 152, 0, 0, 0, 0, 0, 7, 0, 0, 55, 35, 135, 241],[0, 35, 214, 0, 0, 0, 136, 0, 0, 0, 0, 0, 0, 0, 244, 97, 102, 191],[0, 183, 0, 144, 0, 0, 33, 0, 270, 67, 175, 0, 0, 169, 0, 42, 121, 0],[0, 129, 0, 243, 12, 292, 0, 262, 0, 0, 16, 276, 0, 220, 190, 0, 0, 175],[0, 0, 278, 0, 0, 0, 45, 239, 273, 0, 165, 0, 4, 0, 0, 0, 66, 0],[24, 0, 0, 0, 0, 121, 218, 0, 183, 0, 72, 183, 0, 79, 0, 0, 0, 298],[232, 27, 0, 14, 9, 0, 6, 245, 0, 26, 0, 0, 209, 0, 0, 0, 0, 263],[89, 0, 152, 0, 96, 0, 275, 88, 200, 97, 147, 16, 15, 128, 292, 0, 107, 240],[0, 154, 0, 135, 0, 0, 282, 0, 0, 0, 0, 0, 0, 0, 231, 0, 0, 0],[0, 242, 281, 290, 116, 231, 106, 0, 0, 0, 247, 0, 0, 0, 0, 0, 0, 189],[169, 243, 0, 128, 194, 66, 0, 82, 0, 99, 0, 294, 0, 0, 96, 101, 181, 144],[0, 0, 126, 0, 37, 32, 16, 14, 151, 255, 132, 0, 24, 0, 105, 0, 0, 0],[0, 274, 0, 0, 51, 90, 293, 241, 0, 0, 0, 175, 200, 0, 290, 135, 0, 0],[143, 0, 299, 0, 0, 285, 0, 154, 136, 0, 0, 271, 166, 0, 235, 228, 0, 26],[0, 138, 57, 0, 0, 0, 216, 227, 0, 0, 9, 218, 0, 278, 201, 267, 74, 0],[0, 0, 279, 145, 15, 279, 0, 0, 148, 0, 91, 0, 0, 0, 0, 0, 155, 9],[0, 0, 104, 0, 0, 0, 0, 0, 167, 0, 133, 127, 105, 195, 110, 138, 56, 10],[131, 0, 0, 0, 0, 125, 0, 0, 163, 11, 0, 0, 0, 0, 0, 0, 258, 165],[0, 0, 184, 294, 0, 209, 0, 146, 0, 0, 0, 82, 273, 0, 88, 94, 0, 0],[14, 115, 137, 111, 225, 0, 0, 0, 152, 298, 0, 149, 113, 15, 0, 296, 165, 0],[9, 0, 0, 247, 187, 0, 0, 0, 0, 64, 66, 0, 156, 0, 0, 0, 0, 0],[0, 0, 0, 0, 0, 214, 86, 0, 0, 69, 0, 0, 0, 294, 0, 161, 132, 241],[0, 227, 225, 216, 90, 0, 0, 0, 0, 0, 0, 0, 139, 51, 0, 164, 170, 0],[0, 244, 0, 0, 0, 0, 84, 78, 0, 0, 295, 0, 164, 0, 75, 0, 68, 11],[209, 0, 136, 252, 131, 19, 0, 296, 131, 213, 178, 259, 98, 0, 0, 6, 184, 118],[99, 271, 172, 0, 0, 0, 0, 119, 0, 109, 271, 0, 12, 0, 0, 255, 1, 51],[0, 260, 0, 0, 0, 22, 0, 0, 97, 0, 0, 211, 119, 63, 0, 206, 0, 150]]
    prediction = predictions[0]

    
    greedy_general_backpack(machine_number,machine_name,machine_config,flavors_number,flavors_unique,flavors_config,prediction,is_random=True)
    exit()


    for prediction in predictions:
        max_score = None
        best_result = None
        min_count = None
        

        start = datetime.now()
        i = 0
        
        percent = [0.99,0.98]

        while (datetime.now()-start).seconds<50:
            # p = random.choice(percent)
            p = percent[i%len(percent)]  
            # print(p)
            # backpack_count,backpack_result = backpack(machine_number,machine_name,machine_config,flavors_number,flavors_unique,flavors_config,prediction,is_random=True)
            backpack_count,backpack_result = greedy_99_backpack(machine_number,machine_name,machine_config,flavors_number,flavors_unique,flavors_config,prediction,score_treadhold=p)

            cpu_rate,mem_rate = get_backpack_score(machine_number,machine_config,flavors_unique,flavors_config,backpack_result)

            # find the best score solution 
            score  = (cpu_rate+mem_rate)/2.0
            
            # print(i,score)
            i+=1

            if not max_score or max_score<score:
                max_score = score
                best_result = backpack_result
                min_count = backpack_count
        
        backpack_count,backpack_result = random_k_times(machine_number,machine_name,machine_config,flavors_number,flavors_unique,flavors_config,prediction,k=500)
        cpu_rate,mem_rate = get_backpack_score(machine_number,machine_config,flavors_unique,flavors_config,backpack_result)
        # find the best score solution 
        score  = (cpu_rate+mem_rate)/2.0

        print(score)
    
    if not max_score or max_score<score:
        max_score = score
        best_result = backpack_result
        min_count = backpack_count

    
    print("max_score-->",max_score)
    backpack_count = min_count
    backpack_result = best_result


    # --------------build output----------------#

    result = []
    result.append('{}'.format(sum(prediction)))
    for i in range(len(prediction)):
        result.append('flavor{} {}'.format(flavors_unique[i],prediction[i]))

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
