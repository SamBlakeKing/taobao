# -*- coding: utf-8 -*

from scipy import  stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_excel('test-yzh.xlsx')

df.index = df['company']

temp = df.ix[1].fillna(1)
temp['initial'] = 1
temp_E = np.mat(temp.iloc[:,6].as_matrix())

res1 = []
for i in range(371):
    temp = df.ix[i+1].fillna(1)
    temp['initial'] = 1

    xMat = np.mat(temp.iloc[:,(7,3)].as_matrix())
    xMat = np.row_stack((xMat.T, temp_E))
    xMat = xMat.T
    yMat = np.mat(temp.iloc[:,2].as_matrix())

    xTx = xMat.T * xMat
    # print xTx
    if np.linalg.det(xTx) == 0.0:
        res1.append(0)
    else:
        ws = xTx.I * (xMat.T * yMat.T)
        res1.append(float(ws[1]))

res2 = []
for i in range(371):
    temp = df.ix[i+1].fillna(1)
    temp['initial'] = 1

    xMat = np.mat(temp.iloc[:,(7,5)].as_matrix())
    xMat = np.row_stack((xMat.T, temp_E))
    xMat = xMat.T
    yMat = np.mat(temp.iloc[:,4].as_matrix())

    xTx = xMat.T * xMat
    # print xTx
    if np.linalg.det(xTx) == 0.0:
        res2.append(0)
    else:
        ws = xTx.I * (xMat.T * yMat.T)
        res2.append(float(ws[1]))

print res1
print res2

data = pd.DataFrame({'b':res1, 'd':res2})
data.to_excel('temp.xlsx',sheet_name='Sheet1')