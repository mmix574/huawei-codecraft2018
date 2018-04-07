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

    # print([x[1] for x in solution_set])

    for s in solution_set:
        if not minium_count:
            minium_count = s[1]
            best_solution = s
        elif minium_count>s[1]:
            minium_count = s[1]
            best_solution = s
    return best_solution




# class BasePredictor:
#     def __init__(self):
#         self.N = 3

#     def predict(self,X):
#         if shape(X)[0] > self.N:
#             X = X[-self.N:]
#         return X[-1]

#     def fit(self,sample,multi_fit=False,sample_weight=None):
#         self.multi_fit = multi_fit
#         self.sample_weight = sample_weight
#         self.sample = sample

#     def _loss_calc(self,y,y_):
#         # y_ = [round(e) for e in y_]
#         L2_loss = square(minus(y,y_))
#         return L2_loss

#     def _score_calc(self,y,y_):
#         numerator = sqrt(sum(square(minus(y,y_))))
#         denominator = sqrt(sum(square(y))) + sqrt(sum(square(y_)))

#         if denominator==0:
#             return 0
#         else:
#             return 1-(numerator/denominator)

#     def _single_loss(self,sample):
#         datas = self._test_sample(sample)
#         lo = []
#         for d in datas:
#             X,y = d
#             p = self.predict(X)
#             lo.append(self._loss_calc(p,y))
#         losses = mean(lo,axis=1)
#         return losses


#     def _singel_score(self,sample):
#         datas = self._test_sample(sample)
#         s = 0
#         for d in datas:
#             X,y = d
#             p = self.predict(X)
#             s += self._score_calc(p,y)
#         s/=len(datas)
#         return s



#     def loss(self):
#         if self.multi_fit==True:
#             l = []
#             for i in range(len(self.sample)):
#                 l.append(multiply(self._single_loss(self.sample[i]),self.sample_weight[i]))
#             return mean(l,axis=1)
#         else:
#             return self._single_loss(self.sample)

#     def score(self):
#         if self.multi_fit==True:
#             s_l = []
#             for s in self.sample:
#                 s_l.append(self._singel_score(s))
#             return dot(s_l,self.sample_weight)
#             # return mean(s_l)
#         else:
#             return self._singel_score(self.sample)

#     def total_loss(self):
#         return sum(self.loss())

#     # default use 3 slice
#     def _test_sample(self,sample):
#         assert(self.N<len(sample))
#         assert(dim(sample)==2)
#         R = []
#         for i in range(self.N,len(sample)):
#             X = [sample[i-self.N+k] for k in range(self.N)]
#             y = sample[i]
#             R.append((X,y))
#         return R

# class Average(BasePredictor):
#     def __init__(self):
#         BasePredictor.__init__(self)
#         self.N = 3
    
#     def fit(self,sample,multi_fit=False,sample_weight=None):
#         self.multi_fit = multi_fit
#         self.sample_weight = sample_weight
#         self.sample = sample

#     def predict(self,X):
#         if shape(X)[0] > self.N:
#             X = X[-self.N:]
#             return self.predict(X)
#         return X[-1]




# class SimpleFilter(BasePredictor):
#     pass

#     # # not work currently
#     # def sample_filtering(B):
#     #     A = matrix_copy(B)
#     #     # column vector mean,standard deviation
#     #     def mean_std(A):
#     #         var_ = []
#     #         A_T = matrix_transpose(A)
#     #         means = mean(A,axis=1)
#     #         for i in range(shape(A)[1]):
#     #             var_.append(mean((square(minus(A_T[i],means[i])))))
#     #         return means,var_
#     #     means,var_ = mean_std(A)

#     #     for i in range(shape(A)[0]):
#     #         for j in range(shape(A)[1]):
#     #             if abs(A[i][j]-means[j]) > 0.5* var_[j]:
#     #                 A[i][j] = A[i][j]*(1/3.0) + means[j]*(2/3.0)
#     #     return A


# class Coorcoef(BasePredictor):
#     def __init__(self,lower_bound=0.1,upper_bound=0.3):
#         BasePredictor.__init__(self)
#         self.N = 3
#         self.lower_bound = lower_bound
#         self.upper_bound = upper_bound

#     def fit(self,sample,multi_fit=False,sample_weight=None):
#         self.multi_fit = multi_fit
#         self.sample_weight = sample_weight
#         self.sample = sample

#     def predict(self,sample):
#         if shape(sample)[0] > self.N:
#             X = sample[-self.N:]
#             return self.predict(X)

#         from paramater import get_corrcoef_prior
#         prior_corrcoef = get_corrcoef_prior(shape(sample)[1])

#         for i in range(len(prior_corrcoef)):
#             for j in range(len(prior_corrcoef[0])):
#                 if prior_corrcoef[i][j]>self.lower_bound and prior_corrcoef[i][j]<self.upper_bound:
#                     prior_corrcoef[i][j] = 0  

#         X = matrix_matmul(sample,prior_corrcoef)
#         return X[-1]

# class LR(BasePredictor):
#     def __init__(self):
#         BasePredictor.__init__(self)
#         self.N = 3

#     def fit(self,sample,multi_fit=False,sample_weight=None):
#         pass
        
#     def predict(self,sample):
#         pass    