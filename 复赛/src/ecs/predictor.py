import math
import random
import re
from datetime import datetime, timedelta

from learn.ridge import Ridge
from linalg.common import (abs, apply, dim, fancy, mean, minus, reshape, shape,
                           sqrt, sum, zeros)
from linalg.matrix import hstack, stdev
from linalg.vector import arange, argmax, argmin


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
    def outlier_handling(sample,method='mean',max_sigma=4):
        assert(method=='mean')
        std_ = stdev(sample)
        mean_ = mean(sample,axis=0)
        for i in range(shape(sample)[0]):
            for j in range(shape(sample)[1]):
               if sample[i][j]-mean_[j] >max_sigma*std_[j]:
                    if method=='mean':
                        sample[i][j] = mean_[j]
        return sample

    sample = outlier_handling(sample,method='mean')
    # from preprocessing import exponential_smoothing
    # sample = exponential_smoothing(sample)
    # sample = sample[-7:]

    prediction = []
    skip_days = (predict_start-training_end).days
    for i in range(shape(sample)[1]):
        clf = Ridge(alpha=2)
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

    type_i_fix = argmin(abs(minus(machine_rate,cpu_predict/float(mem_prefict))))
    
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
                    mem_prefict -= vm_config['MEM']

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
                mem_prefict -= vm_config['MEM']

                vms.pop(0)
                
                # add @2018-04-18
                # select next type of entity machine
                # type_i = random.choice(range(machine_number))
                if mem_prefict==0:
                    break

                # 1.Greedy Algorithm
                type_i = argmin(abs(minus(machine_rate,cpu_predict/float(mem_prefict))))

                # 2.Simulated annealing Algorithm
                # if random.random()<0.01:
                #     type_i = random.choice(range(machine_number))
                # else:
                #     type_i = argmin(abs(minus(machine_rate,cpu_predict/float(mem_prefict))))
                # good score on local testing set. 

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


def get_approximate_meta_solutions(machine_number,machine_name,machine_config,flavors_number,flavors_unique,flavors_config,prediction,max_iter=1000,score_treadhold=0.8):
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


    def generate_single_prediction_based(config,flavors_unique,flavors_config,vms,max_iter=1000,score_treadhold=0.8):
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
            if score>score_treadhold:
                result.add(solu)
        
        return result

    for i in range(machine_number):
        solu1 = generate_single_prediction_based(machine_config[i],flavors_unique,flavors_config,vms,max_iter=max_iter,score_treadhold=0.99)
        # solu2 = generate_single_expert_based(machine_config[i],flavors_unique,flavors_config,vms,max_iter=max_iter,score_treadhold=0.99)
        
        solu = solu1

        meta_solu.append(solu)

    return meta_solu



def dynamic_programming_backpack(machine_number,machine_name,machine_config,flavors_number,flavors_unique,flavors_config,prediction):
    backpack_result = None
    solutions = get_approximate_meta_solutions(machine_number,machine_name,machine_config,flavors_number,flavors_unique,flavors_config,prediction,max_iter=1000,score_treadhold=0.99)

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
    _,backpack_result_2 = random_k_times(machine_number,machine_name,machine_config,flavors_number,flavors_unique,flavors_config,prediction,k=100)



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
        # backpack_count,backpack_result = dynamic_programming_backpack(machine_number,machine_name,machine_config,flavors_number,flavors_unique,flavors_config,prediction)
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
    
    # debugging
    # dynamic_programming_backpack(machine_number,machine_name,machine_config,flavors_number,flavors_unique,flavors_config,prediction)
    # exit()

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
    for i in range(10):
        # backpack_count,backpack_result = backpack(machine_number,machine_name,machine_config,flavors_number,flavors_unique,flavors_config,prediction,is_random=True)
        backpack_count,backpack_result = dynamic_programming_backpack(machine_number,machine_name,machine_config,flavors_number,flavors_unique,flavors_config,prediction)

        cpu_rate,mem_rate = get_backpack_score(machine_number,machine_config,flavors_unique,flavors_config,backpack_result)

        # find the best score solution 
        score  = (cpu_rate+mem_rate)/2.0
        
        print(i,score)

        if not max_score or max_score<score:
            max_score = score
            best_result = backpack_result
            min_count = backpack_count
    
    print("max_score-->",max_score)

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
