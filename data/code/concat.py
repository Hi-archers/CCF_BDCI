import csv
import json
import numpy as np
import pandas as pd
import random

day = pd.read_csv('../user_data/result/org.csv',encoding = 'utf-8')
local = pd.read_csv('../user_data/result/local.csv',encoding = 'utf-8')
mon = pd.read_csv('../user_data/result/mon.csv',encoding = 'utf-8')
org = pd.read_csv('../user_data/result/org.csv',encoding = 'utf-8')
person = pd.read_csv('../user_data/result/person.csv',encoding = 'utf-8')
year = pd.read_csv('../user_data/result/year.csv',encoding = 'utf-8')
yearday = pd.read_csv('../user_data/result/yearday.csv',encoding = 'utf-8')

L = pd.read_csv('../user_data/result/gen_test_lb.csv',encoding = 'utf-8')
A = pd.read_csv('../user_data/result/all.csv',encoding = 'utf-8')

f = open('../prediction_result/result.csv', 'w', encoding='utf-8',newline = '')
csv_writer = csv.writer(f)
csv_writer.writerow(["id", "ret"])

for i, j in enumerate(L['label']):
    if j == 1:
        tmp = []
        one_year = year.loc[i,'ret']
        one_year = json.loads(one_year)

        one_mon = mon.loc[i,'ret']
        one_mon = json.loads(one_mon)

        one_day = day.loc[i,'ret']
        one_day = json.loads(one_day)

        for k,kk in enumerate(one_day):
            try:
                one_day[k] = int(one_day[k])
            except ValueError:
                one_day[k] = random.randint(1,30)
                one_day[k] = str(one_day[k])
            else:
                one_day[k] = str(one_day[k])

        for k in range(5):
            s = str(one_year[k]+' '+one_mon[k]+' '+one_day[k])
            tmp.append(s)

        csv_writer.writerow([A.loc[i,'id'], json.dumps(tmp)])

    elif j == 2:  # yeardata
        csv_writer.writerow([A.loc[i, 'id'], yearday.loc[i, 'ret']])
    elif j == 3:  # local
        csv_writer.writerow([A.loc[i, 'id'], local.loc[i, 'ret']])
    elif j == 4:  # person
        csv_writer.writerow([A.loc[i, 'id'], person.loc[i, 'ret']])
    elif j == 5:  # person
        csv_writer.writerow([A.loc[i, 'id'], org.loc[i, 'ret']])
    else:
        csv_writer.writerow([A.loc[i, 'id'], A.loc[i, 'ret']])

f.close()