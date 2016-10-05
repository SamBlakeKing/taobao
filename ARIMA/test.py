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

model = sm.tsa.AR(temp)
results_AR = model.fit()
# print len(results_AR.roots)

# 自动拟合
# plt.figure(figsize=(10,4))
# plt.plot(temp,'b',label='CHGPct')
# plt.plot(results_AR.fittedvalues, 'r',label='AR model')
# plt.legend()
# plt.show()

# 模型的检验
# delta  = results_AR.fittedvalues - temp[33:]
# plt.figure(figsize=(10,6))
# plt.plot(delta,'r',label=' residual error')
# plt.legend(loc=0)
# plt.show()

# 拟合优度
# delta  = results_AR.fittedvalues - temp[33:]
# score = 1 - delta.var()/temp[33:].var()
# print score

train = temp[:-25]
test = temp[-25:]
output = sm.tsa.AR(train).fit()
predict = output.predict(5500,5524,dynamic=True)
# print len(predict)

comp = pd.DataFrame()
comp['original'] = test
comp['predict'] = predict
print comp