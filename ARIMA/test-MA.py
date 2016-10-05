# -*- coding: utf-8 -*

from scipy import  stats
import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('testdata.csv')
df.index = pd.to_datetime(df.index)
temp = df['passenger_count']
temp = temp.diff()
temp[0] = 0
temp = np.array(temp)

# 画出序列的ACF, MA的阶次判定
# fig = plt.figure(figsize=(20,5))
# ax1=fig.add_subplot(111)
# fig = sm.graphics.tsa.plot_acf(temp,ax=ax1)
# plt.show()

# 建模
order = (0,25)
train = temp[:-25]
test = temp[-25:]
tempModel = sm.tsa.ARMA(train,order).fit()

# 预测
# delta = tempModel.fittedvalues - train
# score = 1 - delta.var()/train.var()

predicts = tempModel.predict(5500, 5524, dynamic=True)
comp = pd.DataFrame()
comp['original'] = test
comp['predict'] = predicts
comp.plot()
print comp