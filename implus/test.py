import numpy as np
import math

train_day29 = []
offline_candidate_day30 = []
online_cadidate_day31 = []


with open('./fresh_comp_offline/tianchi_fresh_comp_train_user.csv') as f:
    context = f.readlines()
    for line in context:
        line.replace('\n', '')
        array = line.split(',')
        if array[0] == 'user_id':
            continue
        day = int(array[-1].split()[0].split('-')[-1])
        uid = (array[0], array[1], day+1)
        if day == 28:
            train_day29.append(uid)
        if day == 29:
            offline_candidate_day30.append(uid)
        if day == 30:
            online_cadidate_day31.append(uid)

train_day29 = list(set(train_day29))
offline_candidate_day30 = list(set(offline_candidate_day30))
online_cadidate_day31 = list(set(online_cadidate_day31))

ui_buy = {}
ui_dict = [{} for i in range(4)]
with open('./fresh_comp_offline/tianchi_fresh_comp_train_user.csv') as f:
    context = f.readlines()
    for line in context:
        line.replace('\n','')
        array = line.split(',')
        if array[0] == 'user_id':
            continue
        day = int(array[-1].split()[0].split('-')[-1])
        uid = (array[0], array[1], day+1)
        type = int(array[2]) - 1
        if uid in ui_dict[type]:
            ui_dict[type][uid] += 1
        else:
            ui_dict[type][uid] = 1
        if array[2] == '4':
            ui_buy[uid] = 1

X = np.zeros((len(train_day29), 4))
y = np.zeros((len(train_day29), ),)
id = 0
for uid in train_day29:
    last_uid = (uid[0], uid[1], uid[2] -1)
    for i in range(4):
        X[id][i] = math.log1p(ui_dict[i][last_uid] if last_uid in ui_dict[i] else 0)

    y[id] = 1 if uid in ui_buy else 0
    id += 1

pX = np.zeros((len(offline_candidate_day30), 4))
id = 0
for uid in offline_candidate_day30:
    last_uid = (uid[0], uid[1], uid[2]-1)
    for i in range(4):
        pX[id][i] = math.log1p(ui_dict[i][last_uid] if last_uid in ui_dict[i] else 0)
    id += 1

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X, y)

py = model.predict_proba(pX)
npy = []
for a in py:
    npy.append(a[1])
py = npy

lx = zip(offline_candidate_day30, py)
lx = sorted(lx, key=lambda x:x[1], reverse=True)

with open('ans.csv', 'w') as wf:
    wf.write('user_id,item_id\n')
    for item in lx:
        wf.write('%s,%s\n' %(item[0][0],item[0][1]))