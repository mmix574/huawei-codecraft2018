import math
import re
from datetime import datetime

def parse_input_lines(input_lines):
    input_lines = [x.strip() for x in input_lines]

    machine_config = {'cpu':None,'mem':None,'disk':None}
    flavors_number = 0
    flavors_config = {}
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

            flavors_config[f]=(core,mem)
            flavors_unique.append(f)
        elif seg == 3:
            optimized = line.strip()
        elif seg == 4:
            predict_times.append(line.strip())

    predict_start_time = datetime.strptime(predict_times[0], "%Y-%m-%d %H:%M:%S")
    predict_end_time = datetime.strptime(predict_times[1], "%Y-%m-%d %H:%M:%S")

    # sorted flavors_unique @date 03-15
    # important, do not delete
    flavors_unique = sorted(flavors_unique)

    return machine_config,flavors_config,flavors_unique,optimized,predict_start_time,predict_end_time

def parse_ecs_lines(ecs_lines,flavors_unique):
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

    training_start_time = ecs_logs[0][1]
    training_end_time = ecs_logs[len(ecs_logs)-1][1]
    return ecs_logs,training_start_time,training_end_time

# checked
def get_flavors_unique_mapping(flavors_unique):
    mapping_index = {}.fromkeys(flavors_unique)
    c = 0
    for f in flavors_unique:
        mapping_index[f] = c
        c+=1
    return mapping_index


def get_machine_config(flavors_unique,return_normal=True):
    config = {1:(1,1024),2:(1,2048),3:(1,4096),4:(2,2048),5:(2,4096),6:(2,8192),7:(4,4096),8:(4,8192),9:(4,16384),10:(8,8192),11:(8,16384),12:(8,32768),13:(16,16384),14:(16,32768),15:(16,65536)}
    cpu_config,mem_config = [],[]
    for f in flavors_unique:
        cpu_config.append(config[f][0])

        if return_normal:
            mem_config.append(config[f][1]/1024)
        else:
            mem_config.append(config[f][1])

    return cpu_config,mem_config
