import random
from linalg.vector import count_nonezero
from metrics import official_score


# em --> {'1':2,...}
def _get_em_weights_of_cpu_and_mem(em,flavors):
    cpu = 0
    mem = 0
    for k,v in em.items():
        cpu += flavors[k][0]*v
        mem += flavors[k][1]*v
    return cpu,mem

def backpack_scoring(machine_config,flavors,optimized,ems):
    assert(optimized=='CPU' or optimized=='MEM')
    machine_count = len(ems)
    if machine_count==0:
        return 0
    
    total_used = 0
    total = 0 
    if optimized=='CPU':
        total = machine_count * machine_config['cpu']
        for em in ems:
            cpu,men = _get_em_weights_of_cpu_and_mem(em,flavors)
            total_used+=cpu
    elif optimized=='MEM':
        total = machine_count * machine_config['mem']
        for em in ems:
            cpu,men = _get_em_weights_of_cpu_and_mem(em,flavors)
            total_used+=men
    return 0 if total==0 else total_used/float(total)

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
    ems = backpack_list
    return ems,entity_machine_sum

def backpack_random_k_times(machine_config,flavors,flavors_unique,predict,optimized,k=100,verbose=False):
    assert(k>=0)
    solution_set = []
    solution_set.append(backpack(machine_config,flavors,flavors_unique,predict))

    for i in range(k):
        solution_set.append(backpack(machine_config,flavors,flavors_unique,predict,is_random=True))

    best_solution = None
    minium_count = None

    for s in solution_set:
        if not minium_count:
            minium_count = s[1]
            best_solution = s
        elif minium_count>s[1]:
            minium_count = s[1]
            best_solution = s
    
    ems = best_solution[0]
    return best_solution


                        
def _convert_predict_dict_to_vector(predict,flavors_unique):
    result = [predict[f] for f in flavors_unique]
    return result



# todo 
def maximize_score_backpack(machine_config,flavors,flavors_unique,predict,optimized,k=100,verbose=False):
    # def new_em(flavors_unique):
    #     m = {}
    #     m.fromkeys(flavors_unique)
    #     for i in flavors_unique:
    #         m[i] = 0
    #     em = dict(m)
    #     return em
    #     assert(k>=0)
    y = _convert_predict_dict_to_vector(predict,flavors_unique)

    solution_set = []
    solution_set.append(backpack(machine_config,flavors,flavors_unique,predict))

    for i in range(k):
        solution_set.append(backpack(machine_config,flavors,flavors_unique,predict,is_random=True))

    best_solution = None
    minium_count = None

    for s in solution_set:
        if not minium_count:
            minium_count = s[1]
            best_solution = s
        elif minium_count>s[1]:
            minium_count = s[1]
            best_solution = s
    
    ems = best_solution[0]
    return best_solution