import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_validation import ShuffleSplit, cross_val_score
from sklearn.grid_search import GridSearchCV


def score(Y,Y_):
    Y = np.array(Y)
    Y_ = np.array(Y_)
    if(len(np.shape(Y))==1):
        Y = Y.reshape(-1,1)
    if(len(np.shape(Y_))==1):
        Y_ = Y_.reshape(-1,1)

    A = np.power(np.sum((Y-Y_)**2,axis=1),0.5)
    B = np.power(np.sum((Y)**2,axis=1),0.5) + np.power(np.sum((Y_)**2,axis=1),0.5)
    # print(1-np.array(A/B))
    score = 1-np.mean(np.array(A/B))
    # print('-->',1-np.mean(np.array(A/B)))
    print(score)
    return score


def load_data(prefix='../data/',argumentation = True):
    list_of_files_train = ['data_2015_1.txt','data_2015_2.txt','data_2015_3.txt','data_2015_4.txt','data_2015_5.txt']
    list_of_files_validation = ['data_2015_12.txt','data_2016_1.txt']

    lists_of_dataframe_train = [pd.read_table(prefix+list_of_files_train[i],header=None) for i in range(len(list_of_files_train))]
    lists_of_dataframe_validation = [pd.read_table(prefix+list_of_files_validation[i],header=None) for i in range(len(list_of_files_validation))]

    df_train = pd.concat(lists_of_dataframe_train,axis=0,ignore_index=True)
    df_validation = pd.concat(lists_of_dataframe_validation,axis=0,ignore_index=True)

    # step 1 preprocessing
    def _preprocessing(df):
        df.columns = ['uuid','flover_raw','time']
        df['time'] = pd.to_datetime(df['time'])
        df['flover'] = df['flover_raw'].apply(lambda x:int(x[x.find('r')+1:]))
        df['date'] = df['time'].apply(lambda x:pd.datetime.date(x))
        ndf = df[['flover','time','date']]
        ndf.index = df['time']
        return ndf
    df_train = _preprocessing(df_train)
    df_validation = _preprocessing(df_validation)

    # step 2 transforming
    def _transform(df):
        flavors_unique = np.sort(np.array(df['flover'].unique(),dtype=np.int))
        start_date = df['date'][0]
        end_date = df['date'][df.shape[0]-1]
        observation = pd.DataFrame(index=pd.date_range(start_date,end_date),columns=flavors_unique).fillna(0)
        for i in flavors_unique:
            observation[i] = df[df['flover']==i][['flover','date']].groupby('date').count()
        observation = observation.fillna(0)
        return observation
    transform_df_train = _transform(df_train)
    transform_df_validation = _transform(df_validation)

    # remove some flavors not in the training set!
    transform_df_validation = transform_df_validation[transform_df_train.columns]


    # step 3 resampling
    def _resample(df,base=0,frequency='7d'):
        import re
        assert('d' == re.findall('\d+(\w)',frequency)[0])
        day = int(re.findall('(\d+)\w',frequency)[0])
        assert(base<day)

        remain = (df_train.shape[0]-base)%day
        df = df.reset_index()[base:-remain]
        df = df.groupby(df.index//day).sum()
        return df


    # data argumentation of training set
    if argumentation:
        resample_df_train = []
        for i in range(6):
            resample_df_train.append(_resample(transform_df_train,base=i))
        resample_df_train = pd.concat(resample_df_train,axis=0,ignore_index=True)
    else:
        resample_df_train = _resample(transform_df_train)


    # data argumentation of validation set
    if argumentation:
        resample_df_validation = []
        for i in range(6):
            resample_df_validation.append(_resample(transform_df_validation,base=i))
        resample_df_validation = pd.concat(resample_df_validation,axis=0,ignore_index=True)
    else:
        resample_df_validation = _resample(transform_df_validation)

    
    return resample_df_train,resample_df_validation


# columns ==>[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 21, 22, 23]
# df_train,df_validation = load_data()

def load_xy(prefix='../data/',argumentation = True,N=1):
    resample_df_train,resample_df_validation = load_data(prefix=prefix,argumentation=argumentation)

    def XY_bulid(df,N=1):
        X = []
        for i in range(N):
            X.append(df.shift(-i))
        X = np.hstack(X)[:-N]
        y = np.array(df)[N:]
        return X,y
    
    X_train,Y_train = XY_bulid(resample_df_train,N=N)
    X_val,Y_val = XY_bulid(resample_df_validation,N=N)

    return X_train,Y_train,X_val,Y_val    
    

# todo
def load_xy_with_preprocessing(prefix='../data/',argumentation = True,N=1):
    X_train,Y_train,X_val,Y_val = load_xy(prefix=prefix,argumentation=argumentation,N=N)


# load_xy_with_preprocessing(prefix='../data/',argumentation = True,N=2)


