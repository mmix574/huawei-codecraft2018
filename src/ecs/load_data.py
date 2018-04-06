from __future__ import print_function

from predictor import resample
from utils import parse_ecs_lines   
import paramater

from datetime import timedelta
from datetime import datetime


# def resample(ecs_logs,flavors_unique,training_start_time,predict_start_time,frequency='7d',weekday_align=None,N=1,get_flatten=False):


def load_data(flavors_unique,frequency='7d',weekday_align=None,N=1,get_flatten=False,argumentation=False,which=None):
    if which != None:
        if type(which) == int:
            which = [which]
        else:
            pass
    lines_list = [paramater.get_merge_1(),paramater.get_merge_2(),paramater.get_merge_3()]

    if which!=None:
        lines_list = [lines_list[i] for i in range(len(lines_list)) if i in which] 

    X_train,Y_train = [],[]
    for lines in lines_list:
        ecs_logs,training_start_time,training_end_time = parse_ecs_lines(lines)
        d = training_end_time.date() + timedelta(days=1)
        predict_start_time = datetime.combine(d, datetime.min.time())
        X,y,_ = resample(ecs_logs,flavors_unique,training_start_time,predict_start_time,frequency=frequency,weekday_align=weekday_align,N=N,get_flatten=get_flatten,argumentation=argumentation)
        
        X_train.extend(X)
        Y_train.extend(y)
    
    return X_train,Y_train

