import random
from linalg.vector import count_nonezero
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

    def _get_em_weights_of_cpu_and_mem(em,flavors):
        cpu = 0
        mem = 0
        for k,v in em.items():
            cpu += flavors[k][0]*v
            mem += flavors[k][1]*v
        return cpu,mem

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

def backpack_random_k_times(machine_config,flavors,flavors_unique,predict,k=100):
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
    return best_solution


from metrics import official_score

# todo 
def backpack_scoring(machine_config,optimized,backpack_list):
    pass


def maximize_score_backpack(machine_config,optimized,flavors,flavors_unique,predict):
    # entity_machine_sum 5
    # backpack_list  [em,em,em..]
    def new_em(flavors_unique):
        m = {}
        m.fromkeys(flavors_unique)
        for i in flavors_unique:
            m[i] = 0
        em = dict(m)
        return em


    # return backpack_list,entity_machine_sum