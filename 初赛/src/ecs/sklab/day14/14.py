import sys
sys.path.append('..')

from load_data import load_data

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df_train,df_test = load_data('../../data/')




# plt.plot(df_train,marker='o')
# plt.show()
# df_test.to_csv('2.cvs')



