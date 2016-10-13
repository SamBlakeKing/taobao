# -*- coding: utf-8 -*

from scipy import  stats
import statsmodels.api as sm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('result.csv')
df.index = df['slice10min'] # pd.to_datetime(df['slice10min'], format='%Y-%m-%d-%H-%M-%S')
temp = df['07:00:00':'14:50:50']

temp = df['passengerCount']

# temp = temp.diff()
# temp[0] = 0

# 差分处理
data2 = df['07:00:00':'14:50:50']['passengerCount']
data2Diff = data2.diff()


# ---------------------------------

# ADF检验平稳性
t = sm.tsa.stattools.adfuller(temp)
output=pd.DataFrame(index=['Test Statistic Value', "p-value", "Lags Used", "Number of Observations Used","Critical Value(1%)","Critical Value(5%)","Critical Value(10%)"],columns=['value'])
output['value']['Test Statistic Value'] = t[0]
output['value']['p-value'] = t[1]
output['value']['Lags Used'] = t[2]
output['value']['Number of Observations Used'] = t[3]
output['value']['Critical Value(1%)'] = t[4]['1%']
output['value']['Critical Value(5%)'] = t[4]['5%']
output['value']['Critical Value(10%)'] = t[4]['10%']
print output

# ---------------------------------

# 尝试PACF和ACF来判断p、q , result (4, 8)
# length = len(temp)
#
# fig = plt.figure(figsize=(20,10))
# ax1=fig.add_subplot(211)
# fig = sm.graphics.tsa.plot_acf(temp,lags=length - 10,ax=ax1)
# ax2 = fig.add_subplot(212)
# fig = sm.graphics.tsa.plot_pacf(temp,lags=length - 10,ax=ax2)
# plt.show()

# acf_data = np.array(temp)
# print sm.tsa.arma_order_select_ic(acf_data,max_ar=10,max_ma=10,ic='aic')['aic_min_order']  # AIC

# ---------------------------------

# order = (15,3)
# data = np.array(temp) # 差分后，第一个值为NaN
# rawdata = np.array(data2)
# train = data[:-10]
# test = data[-10:]
# model = sm.tsa.ARMA(train,order).fit()

# plt.figure(figsize=(15,5))
# plt.plot(model.fittedvalues,label='fitted value')
# plt.plot(train,label='real value')
# plt.legend(loc=0)
# plt.show()

# 预测最后十组数据
# l = len(train)
# predicts = model.predict(l, l+9, dynamic=True)
# print len(predicts)
# comp = pd.DataFrame()
# comp['original'] = test
# comp['predict'] = predicts
# comp.plot(figsize=(8,5))
# print comp

# 残差
# delta = model.fittedvalues - train
# acf,q,p = sm.tsa.acf(delta,nlags=10,qstat=True)  ## 计算自相关系数 及p-value
# out = np.c_[range(0,l-1), acf[0:], q, p]
# output=pd.DataFrame(out, columns=['lag', "AC", "Q", "P-value"])
# output = output.set_index('lag')
# print output

# rec = [rawdata[-11]]
# pre = model.predict(37, 46, dynamic=True) # 差分序列的预测
# for i in range(10):
#     rec.append(rec[i]+pre[i])
#
# plt.figure(figsize=(10,5))
# plt.plot(rec[-10:],'r',label='predict value')
# plt.plot(rawdata[-10:],'blue',label='real value')
# plt.legend(loc=0)
# plt.show()
