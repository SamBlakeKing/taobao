# -*- coding: utf-8 -*

from scipy import  stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('test.csv')

# 获取样本数据
df.index = df['time']
randIndex = np.arange(900)
xMat = df.iloc[randIndex[:900],1:-1].as_matrix()
yMat = df.iloc[randIndex[:900],-1].as_matrix()

xTestMat = df.iloc[900:,1:-1].as_matrix()
yTestMat = df.iloc[900:,-1].as_matrix()
xTestMat = [[ i+1  for i in j] for j in xTestMat]
xTestMat = np.column_stack((np.linspace(1,1,53),xTestMat))
xTestMat = np.mat(xTestMat)
yTestMat = np.mat(yTestMat)

xMat = [[ i+1  for i in j] for j in xMat]
xMat = np.column_stack((np.linspace(1,1,900),xMat))
xMat = np.mat(xMat)
yMat = np.mat(yMat)

# same things

# 选取几个相关系数较高的特征点
# listChioce = [i for i in range(19) if i not in [7, 8, 9, 10, 11, 17]]
# xTestMat = xTestMat[:,tuple(listChioce)]
# xMat = xMat[:,tuple(listChioce)]

# 标准化
# from sklearn.preprocessing import StandardScaler
# xMat = StandardScaler().fit_transform(xMat)
# yMat = StandardScaler().fit_transform(yMat)


from sklearn.feature_selection import SelectKBest
from scipy.stats import pearsonr
# print np.ndarray(np.asarray(yMat))
# print np.shape(np.ndarray())
# print np.shape(np.ndarray(xMat))
# SelectKBest(lambda X, Y: np.array(map(lambda x:pearsonr(x, Y), X.T)).T, k=2)\
#     .fit_transform(np.array(np.asarray(xMat)), np.array(np.asarray(yMat.T)))

# yMean = np.mean(yMat, 0)
# yMat =  yMat - yMean
# xMean = np.mean(xMat, 0)
# xVar = np.var(xMat, 0)
# xVar[0] = 1
# xMat = (xMat - xMean)/xVar
#
# print xMat - np.mat(xMatt)

# # 计算regression的系数
# xTx = xMat.T * xMat
# denom = xTx + np.eye(np.shape(xMat)[1])*0.2
# xTx = denom
# if np.linalg.det(xTx) == 0.0:
#     print "Can't regress"
# else:
#     ws = xTx.I * (xMat.T * yMat.T)
#     # print ws
#     yTestHat = xTestMat * ws
#     print np.corrcoef(yTestHat.T, yTestMat)
#     # print yTestHat

# 岭回归
# def ridgeRegree(xMat, yMat, lam=0.2):
#     xTx = xMat.T * xMat
#     denom = xTx + np.eye(np.shape(xMat)[1])*lam
#     xTx = denom
#     if np.linalg.det(xTx) == 0.0:
#         print "Can't regress"
#         return
#     else:
#         ws = xTx.I * (xMat.T * yMat.T)
#         return ws
#
# numTestPts = 30
# wMat = np.zeros((numTestPts, np.shape(xMat)[1]))
# for i in range(numTestPts):
#     ws = ridgeRegree(xMat, yMat, np.exp(i-10))
#     wMat[i,:] = ws.T
#
# wMat = wMat[:,1:]
# flg = plt.figure()
# ax = flg.add_subplot(111)
# ax.plot(wMat)
# plt.show()