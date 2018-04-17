# coding=utf-8
from __future__ import print_function
import sys
import os
from datetime import datetime
import math
import re


def mean(A):
    assert(type(A)==list)
    length = len(A)
    _sum = 0
    for i in range(len(A)):
        _sum+=A[i]
    return _sum/float(length)
def squrt(A):
    assert(type(A)==list)
    return [x **2 for x in A]
def minus(A,B):
    assert(type(A)==list and type(B)==list and len(A)==len(B))
    return [A[i] - B[i] for i in range(len(A))]
def plus(A,B):
    assert(type(A)==list and type(B)==list and len(A)==len(B))
    return [A[i] + B[i] for i in range(len(A))]
def root(a):
    assert(a>=0)
    return (math.pow(a,0.5))

def matrix_sum(A,axis=0):
    assert(axis==0)
    assert(type(A)==list and len(A) >0 and type(A[0])==list)
    s = [0 for _ in range(len(A[0]))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            s[j] += A[i][j]
    return s
# -----------------------

def read_lines(file_path):
    if os.path.exists(file_path):
        array = []
        with open(file_path, 'r') as lines:
            for line in lines:
                array.append(line)
        return array
    else:
        print('file not exist: ' + file_path)
        return None

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


# fix bug
# @ 2018-04-11
def parse(inputFile,resultFilePath,testInputFile):
    # step1.   
    input_lines = read_lines(inputFile)
    predict_array = read_lines(resultFilePath)
    test_array = read_lines(testInputFile)

    machine_config,flavors_number,flavors,flavors_unique,optimized,predict_start_time,predict_end_time = parse_input_lines(input_lines)
    ecs_logs,training_start_time,training_end_time = parse_ecs_lines(test_array)
    # parse actual 

    actual = {}.fromkeys(flavors_unique)
    for f in flavors:
        actual[f] = 0
    for f,t in ecs_logs:
        # skip sonme flavors which is not given in the inputFile 
        if f in flavors_unique:
            actual[f] += 1



    # step2.
    # parse predict result
    predict_virtual_machine_count = int(predict_array[0])
    predict = {}
    predict.fromkeys(flavors_unique)
    for f in flavors_unique:
        predict[f] = 0

    for i in range(1,predict_array.index('\n')):
        raw = predict_array[i].split(' ')
        f = int(raw[0][raw[0].find('r')+1:])
        count = int(raw[1])
        predict[f] = count
    # checked

    predict_entity_machine_count = int(predict_array[predict_array.index('\n')+1])
    backpack_list = []
    for i in range(predict_array.index('\n')+2,len(predict_array)):
        match = re.findall(r'r(\d+) (\d+)',predict_array[i])
        em = {}.fromkeys(flavors_unique)
        for f in flavors_unique:
            em[f] = 0
        for k,v in match:
            em[int(k)] = int(v)
        
        backpack_list.append(em)

    return (predict,actual,flavors_unique),(machine_config,optimized,flavors_unique,flavors,backpack_list)

def _partial_score_predict(predict,actual):
    y_ = list(predict.values())
    y = list(actual.values())

    numerator = root(mean(squrt(minus(y,y_))))
    # print(numerator/denominator,1-(numerator/denominator))
    denominator = root(mean(squrt(y)))+root(mean(squrt(y_)))
    if denominator==0:
        return 0
    return 1-(numerator/denominator)

def predict_summary(predict_result):
    print('\n'+'-'*20 +'predict summary below' +'-'*20)
    predict,actual,flavors_unique = [p[0] for p in predict_result],[p[1] for p in predict_result],[p[2] for p in predict_result]

    flavors_unique = flavors_unique[0]
    # comnpute each score of predictions
    scores = [_partial_score_predict(predict[i],actual[i]) for i in range(len(predict))]
    
    if len(scores)==1:
        print(predict[0])
        print(actual[0])

    print('finalscore-->',mean(scores))


    return scores

def _get_em_weights_of_cpu_and_mem(em,flavors):
    cpu = 0
    mem = 0
    for k,v in em.items():
        cpu += flavors[k][0]*v
        mem += flavors[k][1]*v
    return cpu,mem

def backpack_summary(backpack_result):
    # print('\n'+'-'*20 +'backpack summary below' +'-'*20)
    scores_cpu = []
    scores_mem = []
    for each in backpack_result:
        machine_config,optimized,flavors_unique,flavors,backpack_list = each
        total_cpu = 0
        total_mem = 0
        predict_entity_machine_count = len(backpack_list)

        if predict_entity_machine_count==0:
            scores_cpu.append(0)
            scores_mem.append(0)
            continue
        for em in backpack_list:
            cpu,mem = _get_em_weights_of_cpu_and_mem(em,flavors)

            # print('cpu-->',cpu/float(machine_config['cpu']))
            # print('mem-->',mem/float(machine_config['mem']))
            total_cpu += cpu
            total_mem += mem
        assert(optimized=='CPU' or optimized=='MEM')


        score_cpu = total_cpu/float(machine_config['cpu']*predict_entity_machine_count)
        score_mem = total_mem/float(machine_config['mem']*predict_entity_machine_count)

        scores_cpu.append(score_cpu)
        scores_mem.append(score_mem)


    # print('backpack_cpu_score-->',sum(scores_cpu)/float(len(scores_cpu)))
    # print('backpack_mem_score-->',sum(scores_mem)/float(len(scores_mem)))

    return None


if __name__ == '__main__':
    if len(sys.argv)>=2:
        print(sys.argv)
    if sys.argv[1]=='train':
        backtest_folder = 'local_data_2015_01_2015_05/slicing_50days_7days_strike7'

    elif sys.argv[1]=='test':
        backtest_folder = 'local_data_2015_12_2016_01/slicing_50days_7days_strike3'
    elif sys.argv[1]=='val':
        # python27_path = "C:\Python27\python.exe"
        os.system('python ecs.py TrainData_2015.1.1_2015.2.19.txt input_15flavors_cpu_7days.txt output.txt')
        # get parsed  
        p,b = parse('input_15flavors_cpu_7days.txt','output.txt','TestData_2015.2.20_2015.2.27.txt')

        predict_summary([p])
        backpack_summary([b])
        exit()
    else:
        print('command error')
        exit()       
    # python27_path = "C:\Python27\python.exe"
    python27_path = "python"
    dirs = [int(x) for x in os.listdir(backtest_folder)]
    dirs = sorted(dirs)
    if len(sys.argv)>2:
        number = int(sys.argv[2])
        dirs = [d for d in dirs if d==number]

    import random
    temp_folder = 'temp{}'.format(random.random())

    if os.path.exists(temp_folder):
        # os.removedirs(temp_folder)
        import shutil
        shutil.rmtree(temp_folder)

    # create temp folder for backtesting
    os.mkdir(temp_folder)
    for i in dirs:
        os.mkdir('{}/{}'.format(temp_folder,str(i)))

    # build input file
       # like input_5flavors_cpu_7days.txt
    _raw_input_header = ['56 128 1200\n',
 '\n',
 '5\n',
'flavor1 1 1024\n',
'flavor2 1 2048\n',
'flavor3 1 4096\n',
'flavor4 2 2048\n',
'flavor5 2 4096\n',
'flavor6 2 8192\n',
'flavor7 4 4096\n',
'flavor8 4 8192\n',
'flavor9 4 16384\n',
'flavor10 8 8192\n',
'flavor11 8 16384\n',
'flavor12 8 32768\n',
'flavor13 16 16384\n',
'flavor14 16 32768\n',
'flavor15 16 65536\n',
 '\n',
 'CPU\n',
 '\n',
]
    # result of batch testing 
    for i in dirs:

        # write input files
        input_buliding_file = [x for x in _raw_input_header]
        f = open('{}/{}/{}'.format(backtest_folder,str(i),'test_timerange.txt'))
        input_buliding_file.extend(f.readlines())
        f.close()
        f = open('{}/{}/{}'.format(temp_folder,str(i),'t_input_5flavors_cpu_7days.txt'),'w+')
        f.writelines(input_buliding_file)
        f.close()
        # print(input_buliding_file)

        # paths 
        input_file_path = '{}/{}/{}'.format(temp_folder,str(i),'t_input_5flavors_cpu_7days.txt')
        esc_logs_path = '{}/{}/{}'.format(backtest_folder,str(i),'train.txt')
        output_file_path = '{}/{}/{}'.format(temp_folder,str(i),'t_output.txt')

        print(i)
        # execute command
        # print('command\n','{} {} {} {} {}'.format(python27_path,'ecs.py',esc_logs_path,input_file_path,output_file_path))
        from datetime import datetime

        
        strat_time = datetime.now()
        os.system('{} {} {} {} {}'.format(python27_path,'ecs.py',esc_logs_path,input_file_path,output_file_path))
        print('timeused-->',(datetime.now() - strat_time).seconds,'S')
        
        testing_file_path = '{}/{}/{}'.format(backtest_folder,str(i),'test.txt')
        if not (os.path.exists(output_file_path) and os.path.exists(testing_file_path)):
            print('bad~')
            continue
        # assert(os.path.exists(output_file_path) and os.path.exists(testing_file_path))

        # get parsed  
        predict_,backpack_ = parse(input_file_path,output_file_path,testing_file_path)

        predict_summary([predict_])
        backpack_summary([backpack_])

    # delete the temporary file dir
    if os.path.exists(temp_folder):
        import shutil
        shutil.rmtree(temp_folder)





