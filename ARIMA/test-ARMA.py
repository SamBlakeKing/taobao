# -*- coding: utf-8 -*

from scipy import  stats
import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('result.csv')
df.index = pd.to_datetime(df['slice10min'], format='%Y-%m-%d-%H-%M-%S')
temp = df['passengerCount']
# temp = temp.diff()
# temp[0] = 0
temp = np.array(temp)

# acf和pacf的阶预测
# fig = plt.figure()
# ax1=fig.add_subplot(211)
# fig = sm.graphics.tsa.plot_acf(temp,lags=10,ax=ax1)
# plt.ylim(0,1)
#
# ax2 = fig.add_subplot(212)
# fig = sm.graphics.tsa.plot_pacf(temp,lags=10,ax=ax2)
# plt.ylim(0,1)
# plt.show()

# 信息准则定阶
# print sm.tsa.arma_order_select_ic(temp,max_ar=15,max_ma=15,ic='aic')['aic_min_order']
# print sm.tsa.arma_order_select_ic(temp,max_ar=10,max_ma=3,ic='bic')['bic_min_order']  # BIC
# print sm.tsa.arma_order_select_ic(temp,max_ar=10,max_ma=10,ic='hqic')['hqic_min_order'] # HQIC

# # 模型的建立及预测
order = (2,1)
train = temp[:-10]
test = temp[-10:]
tempModel = sm.tsa.ARMA(train,order).fit()

#
# # delta = tempModel.fittedvalues - train
# # score = 1 - delta.var()/train.var()
# # print score
#
# predicts = tempModel.predict(38, 47, dynamic=True)
# print len(predicts)
# comp = pd.DataFrame()
# comp['original'] = test
# comp['predict'] = predicts
# comp.plot()
# print comp

# comp = pd.DataFrame()
# comp['original'] = train
# comp['predict'] = tempModel.fittedvalues
# comp.plot()
# print comp