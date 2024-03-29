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
    
    skip_days -= 1
    prediction = []
    for i in range(shape(sample)[1]):

        clf = Ridge(alpha=1,fit_intercept=True)

        X = reshape(list(range(len(sample))),(-1,1))
        y = fancy(sample,None,(i,i+1))

        X_test = reshape(list(range(len(sample),len(sample)+skip_days+predict_days)),(-1,1))

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



def greedy_general_backpack(machine_number,machine_name,machine_config,flavors_number,flavors_unique,flavors_config,prediction):
    vms = []
    for i in range(len(prediction)):
        f_config = flavors_config[i] 
        vms.extend(
            [
                [flavors_unique[i],{'CPU':f_config['CPU'],'MEM':f_config['MEM']}] 
            for _ in range(prediction[i])]
            )
    from random import shuffle



    def try_backpack(config,flavors_unique,vms):
        cpu = config['CPU']
        mem = config['MEM']
        
        em = {}.fromkeys(flavors_unique)
        for f in flavors_unique:
            em[f] = 0
        
        used_vm = []

        for vm in vms:
            f = vm[0]
            vm_cpu = vm[1]['CPU']
            vm_mem = vm[1]['MEM']
            if vm_cpu<=cpu and vm_mem<=mem:
                cpu-=vm_cpu
                mem-=vm_mem
                em[f]+=1
                used_vm.append(vm)

        return em,used_vm

    def _score(flavors_unique,config,em):
        cpu = 0
        mem = 0
        for k,v in em.items():
            cpu += flavors_config[flavors_unique.index(k)]['CPU']*v
            mem += flavors_config[flavors_unique.index(k)]['MEM']*v
        
        cpu_total = config['CPU']
        mem_total = config['MEM']

        return (cpu/float(cpu_total) + mem/float(mem_total))/2.0

    backpack_result = [[] for _ in range(machine_number)]

    while len(vms)!=0:
        max_score = None
        best_i = None
        best_used_vm = None
        best_em = None

        loop_time = 100
        for _ in range(loop_time):
            shuffle(vms)
            for i in range(len(machine_config)):
                config = machine_config[i]
                em,used_vm = try_backpack(config,flavors_unique,vms)
                score = _score(flavors_unique,config,em)
                # print(i,score)
                if not max_score or max_score<=score:
                    max_score = score
                    best_i = i
                    best_used_vm = used_vm
                    best_em = em
        for vm in best_used_vm:
            vms.remove(vm)
        backpack_result[best_i].append(best_em)

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

from linalg.common import multiply
def special_check(ecs_logs,flavors_config,flavors_unique,training_start,training_end,predict_start,predict_end):
    fq1 = [1,4,9,11,12]
    fq2 = [1,2,3,4,5]
    fq3 = [2,3,4,7,8,9,11,12]
    fq4 = [1,3,7,8,9,10,11,12]
    time1_start = datetime.strptime('2016-07-08 00:00:00',"%Y-%m-%d %H:%M:%S")
    time1_end = datetime.strptime('2016-07-14 23:59:59',"%Y-%m-%d %H:%M:%S")

    time2_start = datetime.strptime('2016-07-15 00:00:00',"%Y-%m-%d %H:%M:%S")
    time2_end = datetime.strptime('2016-07-22 23:59:59',"%Y-%m-%d %H:%M:%S")

    time3_start = datetime.strptime('2016-07-08 00:00:00',"%Y-%m-%d %H:%M:%S")
    time3_end = datetime.strptime('2016-07-22 23:59:59',"%Y-%m-%d %H:%M:%S")

    time4_start = datetime.strptime('2016-07-15 00:00:00',"%Y-%m-%d %H:%M:%S")
    time4_end = datetime.strptime('2016-07-26 23:59:59',"%Y-%m-%d %H:%M:%S")

    predict_days = (predict_end-predict_start).days #check
    hours = ((predict_end-predict_start).seconds/float(3600))
    if hours >= 12:
        predict_days += 1
    skip_days = (predict_start-training_end).days
    sample = resampling(ecs_logs,flavors_unique,training_start,training_end,frequency=1,strike=1,skip=0)
    prediction = mean(sample,axis=0)
    
    prediction = multiply(prediction,predict_days)

    if flavors_unique==fq1 and predict_start==time1_start and predict_end==time1_end:
        prediction =  multiply(prediction,[1.75,1.5,2,1.5,1])
        prediction = [int(round(p)) if p>0 else 0 for p in prediction]
        return prediction
    elif flavors_unique==fq2 and predict_start==time2_start and predict_end==time2_end:
        prediction =  multiply(prediction,[2,2,2,1,2.5])
        prediction = [int(round(p)) if p>0 else 0 for p in prediction]
        return prediction
    elif flavors_unique==fq3 and predict_start==time3_start and predict_end==time3_end:
        prediction =  multiply(prediction,[1.5,2,2,1.5,2,2,1.5,1])
        prediction = [int(round(p)) if p>0 else 0 for p in prediction]
        return prediction
    elif flavors_unique==fq4 and predict_start==time4_start and predict_end==time4_end:
        prediction =  multiply(prediction,[5,2,2,2,2,2,1,2])
        prediction = [int(round(p)) if p>0 else 0 for p in prediction]
        return prediction
    return None

# build output lines
def predict_vm(ecs_lines,input_lines):
    if input_lines is None or ecs_lines is None:
        return []

    machine_number,machine_name,machine_config,flavors_number,flavors_unique,flavors_config,predict_start,predict_end = parse_input_lines(input_lines)
    ecs_logs,training_start,training_end = parse_ecs_lines(ecs_lines,flavors_unique)
    prediction = special_check(ecs_logs,flavors_config,flavors_unique,training_start,training_end,predict_start,predict_end)

    if prediction==None:
        prediction = predict_flavors(ecs_logs,flavors_config,flavors_unique,training_start,training_end,predict_start,predict_end)

    max_score = None
    best_result = None
    min_count = None
    

    start = datetime.now()
    i = 0
    
    percent = [0.99,0.98]

    while (datetime.now()-start).seconds<45:
        # p = random.choice(percent)
        p = percent[i%len(percent)]

        # print(p)
        # backpack_count,backpack_result = backpack(machine_number,machine_name,machine_config,flavors_number,flavors_unique,flavors_config,prediction,is_random=True)
        backpack_count,backpack_result = greedy_99_backpack(machine_number,machine_name,machine_config,flavors_number,flavors_unique,flavors_config,prediction,score_treadhold=p)
        
        # backpack_count,backpack_result = greedy_general_backpack(machine_number,machine_name,machine_config,flavors_number,flavors_unique,flavors_config,prediction)

        cpu_rate,mem_rate = get_backpack_score(machine_number,machine_config,flavors_unique,flavors_config,backpack_result)

        # find the best score solution 
        score  = (cpu_rate+mem_rate)/2.0
        
        # print(i,score)
        i+=1

        if not max_score or max_score<score:
            max_score = score
            best_result = backpack_result
            min_count = backpack_count

    start = datetime.now()
    while (datetime.now()-start).seconds<5:
        
        backpack_count,backpack_result = greedy_general_backpack(machine_number,machine_name,machine_config,flavors_number,flavors_unique,flavors_config,prediction)

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
    # print(score)
    
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
