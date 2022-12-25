import json
import spacy
import pandas as pd
import numpy as np
import os
import sys
path=os.path.dirname(os.path.abspath(__file__))  
sys.path.append(path) 
import time
from utlis import check_mon, check_num,check_date_year_temperature,save_different_category_data,process_temp_test

origin_train_path = '../raw_data/KCT_train_public.txt'
origin_test_path = '../raw_data/KCT_test_public.txt'
base_path = '../user_data/train_data'

num_multi_hop = 0
num_fact = 0
num_common_sense = 0

category_dict = {'other': 0, 'date': 1, 'yearday': 2, 'local': 3, 'person': 4, 'org': 5}
rev_category_dict = {0: 'other', 1: 'date', 2: 'yearday', 3:'local', 4:'person', 5: 'org'}

n = 0
real_ent = 0
other = 0
nlp = spacy.load("en_core_web_sm")

# 字典类型,存放各个类型的数据存在的数量
ent_type = {}
no_ent = []
yes_ent = []

other = []
date = []
yearday = []
local = []
person = []
org = []

OTHER, DATE, YEARDAY, GPE, PERSON, ORG = 0,0,0,0,0,0

labels = []
text = []

# [0:其他, 1:年月日, 2:年, 3:国家, 4:人名, 5:组织名]


with open(origin_train_path) as f:
    for line in f:
        one_data = json.loads(line)
        
        flag = 0
        if one_data['NeedReasoning']:
            num_multi_hop += 1
        if one_data['domain'] == 'Facts':
            num_fact += 1
        if one_data['domain'] == 'Common Sense':
            num_common_sense += 1
        
        if check_date_year_temperature(one_data['query'], one_data['answer'][0])==1:
            flag = 1 
            date.append(line)
            DATE += 1 
            cur_label = 1
            cur_text = one_data['query'].replace('[MASK]', "")
            labels.append(cur_label)
            text.append(cur_text)
            continue

        if check_date_year_temperature(one_data['query'], one_data['answer'][0]) == 2:
            flag = 1 
            yearday.append(line)
            YEARDAY += 1 
            cur_label = 2
            cur_text = one_data['query'].replace('[MASK]', "")
            labels.append(cur_label)
            text.append(cur_text)
            continue

        
        if one_data['domain'] == 'Facts' or one_data['domain'] == 'Common Sense':
            sentence = one_data['query'].replace('[MASK]', one_data['answer'][0])
            doc = nlp(sentence)
            flag = 0
            for ent in doc.ents:
                # print("ent: ",ent)
                if str(ent) in one_data['answer'][0] or one_data['answer'][0] in str(ent):
                    ent_type[ent.label_] = ent_type[ent.label_] + 1 if ent.label_ in ent_type else 1
                    if ent.label_ == 'GPE':
                        flag = 1
                        local.append(line)
                        GPE += 1 
                        cur_label = 3
                        cur_text = one_data['query'].replace('[MASK]', "")

                    elif ent.label_ == 'PERSON':
                        flag = 1
                        person.append(line)    
                        PERSON += 1 
                        cur_label = 4 
                        cur_text = one_data['query'].replace('[MASK]', "")

                    elif ent.label_ == 'ORG':
                        flag = 1
                        org.append(line)
                        ORG += 1
                        cur_label = 5
                        cur_text = one_data['query'].replace('[MASK]', "")
                    
                    if flag == 1:
                        break
            
            if flag == 1:
                yes_ent.append(line)
                real_ent += 1 

        if flag == 0:
            OTHER += 1 
            other.append(line)
            n += 1
            cur_label = 0
            cur_text = one_data['query'].replace('[MASK]', "")

        labels.append(cur_label)
        text.append(cur_text)


for i in range(len(category_dict)):
    if i == 0:
        save_different_category_data(base_path, 'other', other)
        time.sleep(1)

    elif i == 1:
        save_different_category_data(base_path, 'date', date)
        time.sleep(1)

    elif i == 2:
        save_different_category_data(base_path, 'yearday', yearday)
        time.sleep(1)

    elif i == 3:
        save_different_category_data(base_path, 'local', local)
        time.sleep(1)

    elif i == 4:
        save_different_category_data(base_path, 'person', person)
        time.sleep(1)

    elif i == 5:
        save_different_category_data(base_path, 'org', org)
        time.sleep(1)

print("保存6种类别的数据成功,数据预处理还在继续,请稍候...")

Dict = pd.DataFrame({
    'labels':labels,
    'text':text,
})


print("保存的这里的shape: ",Dict.shape,len(labels),len(text))

sum_cnt = len(labels)
train_cnt = int(sum_cnt * 0.9)
val_cnt = sum_cnt - train_cnt
test_cnt = val_cnt

train_Dict = Dict[:train_cnt]
val_Dict = Dict[train_cnt:train_cnt+val_cnt]
test_Dict = val_Dict 


# Dict.to_csv(f'{base_path}/KCT_train_label.csv',index = False,encoding = 'utf-8')
train_Dict.to_csv(f'{base_path}/bert_train.csv',index = False,encoding = 'utf-8')
val_Dict.to_csv(f'{base_path}/bert_val.csv',index = False,encoding = 'utf-8')
test_Dict.to_csv(f'{base_path}/bert_test.csv',index = False,encoding = 'utf-8')
process_temp_test(origin_test_path, f'{base_path}/no_use_temp_test_1.csv', f'{base_path}/no_use_temp_test_2.csv')
# print("Dict: ",train_Dict.shape,val_Dict.shape,test_Dict.shape,"和: ",train_cnt,val_cnt,train_cnt+val_cnt)

print("Data preprocessing finished...")
