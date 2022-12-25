import csv
import json
import numpy as np
import pandas as pd


year = []
mon = []
day = []
with open('../user_data/train_data/date_val.txt',encoding = 'utf-8') as f:
    for line in f:
        one_data = json.loads(line)

        tmp = one_data['answer']
        tmp = tmp[0]
        tmp = tmp.split(' ')

        if int(tmp[0]) < int(tmp[2]):
            one_data['answer'] = [tmp[2]]
            year.append(json.dumps(one_data))

            one_data['answer'] = [tmp[1]]
            mon.append(json.dumps(one_data))

            one_data['answer'] = [tmp[0]]
            day.append(json.dumps(one_data))
        else :
            one_data['answer'] = [tmp[0]]
            year.append(json.dumps(one_data))

            one_data['answer'] = [tmp[1]]
            mon.append(json.dumps(one_data))

            one_data['answer'] = [tmp[2]]
            day.append(json.dumps(one_data))



with open('../user_data/train_data/year_val.txt','w') as f:
    for i in year:
        f.writelines(i+'\n')

with open('../user_data/train_data/mon_val.txt','w') as f:
    for i in mon:
        f.writelines(i+'\n')

with open('../user_data/train_data/day_val.txt','w') as f:
    for i in day:
        f.writelines(i+'\n')
