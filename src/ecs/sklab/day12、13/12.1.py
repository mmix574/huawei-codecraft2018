import matplotlib.pyplot as plt
from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import ShuffleSplit

from sklearn.grid_search import GridSearchCV

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_data(prefix='../data/'):
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

    argumentation = True

    # data argumentation of training set
    if argumentation:
        resample_df_train = []
        for i in range(7):
            resample_df_train.append(_resample(transform_df_train,base=i))
        resample_df_train = pd.concat(resample_df_train,axis=0,ignore_index=True)
    else:
        resample_df_train = _resample(transform_df_train)

    resample_df_validation = _resample(transform_df_validation)
    
    return resample_df_train,resample_df_validation

# columns ==>[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17, 18, 21, 22, 23]
df_train,df_validation = get_data()

def data_target_bulid(df,N=1):
    assert(N>0)
    y = df[N:]
    T = []
    for i in range(N):
        T.append(df.shift(-i))
    X = np.hstack(T)[:-N]
    return np.array(X),np.array(y)

X_train,Y_train = data_target_bulid(df_train,N=1)
X_val,Y_val = data_target_bulid(df_validation,N=1)


from sklearn.preprocessing import StandardScaler
 
StandardScaler().fit_transform(X_train,Y_train)

def score(Y,Y_):
    Y = np.array(Y)
    Y_ = np.array(Y_)
    A = np.power(np.sum((Y-Y_)**2,axis=1),0.5)
    B = np.power(np.sum((Y)**2,axis=1),0.5) + np.power(np.sum((Y_)**2,axis=1),0.5)
    # print(1-np.array(A/B))
    score = 1-np.mean(np.array(A/B))
    # print('-->',1-np.mean(np.array(A/B)))
    print(score)
    return score
p = (np.array(df_validation)[:-1])

# print(score(p,Y_val))

# from sklearn.linear_model import Lasso

# for i in [x/100 for x in range(500)]:
#     clf = Lasso(alpha=i)
#     clf.fit(X_train,Y_train)
#     p = clf.predict(X_val)
#     print(score(p,Y_val))


# from sklearn.tree import DecisionTreeRegressor
# clf = DecisionTreeRegressor()
# clf.fit(X_train,Y_train)
# p = clf.predict(X_val)
# print(score(p,Y_val))


# from sklearn.feature_extraction import 
def data_target_bulid_one_flavor(df,i,N=1):
    d = np.array(df)[:,i].reshape(-1,1)
    L = []
    for i in range(N):
        L.append(np.roll(d,-i))
    X = np.hstack(L)[:-N]
    y = d[N:]
    return X,y    

print(df_train.shape)
for i in range(df_train.shape[1]):
    print(i)
    X,y = data_target_bulid_one_flavor(df_train,i,N=7)
    from sklearn.model_selection import train_test_split

    X_train,X_val,Y_train,Y_val = train_test_split(X,y,random_state=42)

    from sklearn.linear_model import LinearRegression,Lasso

    clf = Lasso(fit_intercept=None)
    clf.fit(X_train,Y_train)
    p = clf.predict(X_val).reshape(-1,1)

    score(p,Y_val)

