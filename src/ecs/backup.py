def predict_flavors_unique_ma(ecs_logs,flavors_unique,training_start_time,training_end_time,predict_start_time,predict_end_time):
    predict = {}.fromkeys(flavors_unique)
    for f in flavors_unique:
        predict[f] = 0
    virtual_machine_sum = 0
    
    if len(ecs_logs) == 0:
        return predict,virtual_machine_sum 

    predict_days = (predict_end_time-predict_start_time).days
    training_days = (training_end_time - training_start_time).days

    _double11 = datetime.strptime('{}-11-11 00:00:00'.format(predict_start_time.year), "%Y-%m-%d %H:%M:%S")
    _double12 = datetime.strptime('{}-12-12 00:00:00'.format(predict_start_time.year), "%Y-%m-%d %H:%M:%S")

    max_ma = training_days//predict_days
    if max_ma ==0:
        return predict,virtual_machine_sum 

    ma = 1
    assert(ma<=max_ma) #passed

    for i in flavors_unique:
        c = [True if flavor==i and (training_end_time-ecs_time).days<predict_days*ma else False for flavor,ecs_time in ecs_logs]
        
        if(predict_start_time<_double12 and predict_end_time>_double12 or predict_start_time<_double11 and predict_end_time>_double11):
            # print('special time')
            scale_time = 5
            predict[i] += int(round(count_nonezero(c)/ma)) * scale_time
            virtual_machine_sum += int(round(count_nonezero(c)/ma)) *scale_time
        else:
            predict[i] = int(round(count_nonezero(c)/ma)) 
            virtual_machine_sum += int(round(count_nonezero(c)/ma)) 

    return predict,virtual_machine_sum

    # modify @ 2018-03-15 
    predict = {}.fromkeys(flavors_unique)
    for f in flavors_unique:
        predict[f] = 0
    virtual_machine_sum = 0
    #end modify
    predict_days = (predict_end_time-predict_start_time).days
    X,y = resample(ecs_logs,flavors_unique,training_start_time,predict_start_time,frequency='{}d'.format(predict_days),N=1,get_flatten=True)

    # from load_data import load_data
    # old_samples = load_data(flavors_unique,frequency='{}d'.format(predict_days),weekday_align=predict_start_time)
    # # old_samples = load_data(flavors_unique,frequency='{}d'.format(predict_days))
    
    # samples = old_samples
    # samples.append(new_sample)
    # weights = [0.1,0.2,0.3,0.4]


    print('hello world')
    exit()
    # weights = [0.1,0.2,0.2,0.5]
    # weights = [0.0,0.2,0.4,0.4]
    # weights = [0.25,0.25,0.25,0.25]
    # weights = [0,0,0,1]


    # m = Merger(samples,weights)
    # not really work on current dataset
    # sample = sample_filtering(sample)

    # from paramater import get_merge_1
    # print(get_merge_1())
    
    # sample_smoothing = mean_smoothing(sample_smoothing,N=3)
    # transeform_sample = matrix_matmul(sample,prior_corrcoef)

    av = Average()
    av.fit(samples,multi_fit=True,sample_weight=weights)

    # print(av.loss())
    # print(av.total_loss())
    # print(av.score())

    
    # p = grid_search(Smoothing,{'w':[0.2,0.25,0.28,0.3,0.35,0.38,0.4]},samples,weights)
    p = grid_search(Smoothing,{'w':arange(0.1,0.7,1000),'N':[5]},samples,weights)
    
    # p = grid_search(Smoothing,{'w':arange(0.1,0.8,1000),'N':[x for x in range(2,shape(new_sample)[0])]},samples,weights,verbose=True)

    sm = Smoothing(**p)
    sm.fit(samples,multi_fit=True,sample_weight=weights)

    # print(sm.loss())
    # print(sm.total_loss())
    # print(sm.score())

    # p = grid_search(Coorcoef,{'lower_bound':arange(-0.2,0.8,50),'upper_bound':arange(-0.2,0.8,50)},samples,weights,verbose=False)

    # co = Coorcoef(**p)
    # co.fit(samples,multi_fit=True,sample_weight=weights)

    _predict_ = sm.predict(new_sample)
    # _predict_ = co.predict(new_sample)

    for f in flavors_unique:
        predict[f] = int(round(_predict_[mapping_index[f]]))
        virtual_machine_sum += int(round(_predict_[mapping_index[f]]))
    return predict,virtual_machine_sum
