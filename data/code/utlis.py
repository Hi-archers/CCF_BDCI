import json
import torch
from torch.utils.data import Dataset
import numpy as np
import random
import pandas as pd
import csv
import time
from tqdm import tqdm
from evaluate import f1_score

Mon = [
    'January',
    'February',
    'March',
    'April',
    'May',
    'June',
    'July',
    'August',
    'September',
    'October',
    'November',
    'December'
]


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def check_mon(answers):
    for mon in Mon:
        if mon in answers:
            return True

def check_num(answers):
    for num in range(0, 10):
        if str(num) in answers:
            flag_num = True
            return True


def check_date_year_temperature(query, answers):
    
    if len(answers.split(' ')) == 3: # 标准年-月-日的标准日期类型数据
        flag_mon = check_mon(answers)
        flag_num = check_num(answers)
        if flag_mon and flag_num:
            return 1

    elif 'year' in query and len(answers)<=4:   # 标准year类型数据
        return 2

    return 0



def save_train_val_file(Dict, base_path, category,phase):

    Dict.to_csv(f'{base_path}/no_use_tmp_{category}_{phase}.csv',index = False,encoding = 'utf-8')
    CSV = open(f'{base_path}/no_use_tmp_{category}_{phase}.csv','r',encoding = 'utf-8')
    output_file = f'{base_path}/{category}_{phase}.txt'

    # print("CSV: ",Dict.shape)
    FILE = open(output_file,'w', encoding='utf-8')

    reader = csv.DictReader(CSV)

    for row in reader:
        row['answer'] = [row['answer']]
        print(json.dumps(row),file=FILE)



def save_different_category_data(base_path, category, lines):
    
    sum_cnt = len(lines)
    train_cnt = int(sum_cnt * 0.9)
    val_cnt = sum_cnt - train_cnt

    qid = []
    query = []
    answer = []
    domain = []
    NeedReasoning = []

    time.sleep(1)

    train_lines = lines[:train_cnt]
    val_lines = lines[train_cnt:]

    phase = "train"
    for line in train_lines:
        tmp = json.loads(line)
        ans = tmp['answer']
        for i in ans:
            qid.append(tmp['qid'])
            query.append(tmp['query'])
            answer.append(i)
            domain.append(tmp['domain'])
            NeedReasoning.append(tmp['NeedReasoning'])

    train_Dict = pd.DataFrame({
        'qid':qid,
        'query':query,
        'answer':answer,
        'domain':domain,
        'NeedReasoning':NeedReasoning
    })
    save_train_val_file(train_Dict, base_path, category, phase)
    

    qid.clear()
    query.clear()
    answer.clear()
    domain.clear()
    NeedReasoning.clear()
    
    phase = "val"
    for line in val_lines:
        tmp = json.loads(line)
        ans = tmp['answer']
        for i in ans:
            qid.append(tmp['qid'])
            query.append(tmp['query'])
            answer.append(i)
            domain.append(tmp['domain'])
            NeedReasoning.append(tmp['NeedReasoning'])
    
    val_Dict = pd.DataFrame({
        'qid':qid,
        'query':query,
        'answer':answer,
        'domain':domain,
        'NeedReasoning':NeedReasoning
    }) 
    save_train_val_file(val_Dict, base_path, category, phase)

    # print("x y: ",x,y,train_Dict.shape,val_Dict.shape)


# 处理真正的test测试集
def process_temp_test(test_path, out_path_1, out_path_2):
    labels = []
    text = []
    with open(test_path,'r',encoding = 'utf-8') as f:
        for line in f:
            one_data = json.loads(line)
            one_data['query'] = one_data['query'].replace('[MASK]', "")
            text.append(one_data['query'])
            labels.append(0)

    Dict_1 = pd.DataFrame({
        'labels':labels,
        'text':text,
    })
    Dict_1.to_csv(out_path_1,index = False,encoding = 'utf-8')
    
    qid = []
    query = []
    label = []
    with open(test_path,'r',encoding = 'utf-8') as f:
        for line in f:
            tmp = json.loads(line)
            qid.append(tmp['qid'])
            query.append(tmp['query'])
            label.append(0)

    Dict_2 = pd.DataFrame({
        'qid':qid,
        'query':query,
        'label':label,
    })
    Dict_2.to_csv(out_path_2,index = False,encoding = 'utf-8')

class mask_dataset(Dataset):

    def __init__(self, data_path, tokenizer,test=False):
        self.data = []
        self.tokenizer = tokenizer
        self.test = test
        i = 0
        with open(data_path) as f:
            for line in f:
                one_data = json.loads(line)
                one_data['query'] = one_data['query'].replace('[MASK]', '<extra_id_0>')

                if  self.test:
                    self.data.append(one_data)
                else:
                    for answer in one_data['answer']:
                        self.data.append(one_data.copy())
                        #self.data[-1]['answer'] = '<extra_id_0> ' + answer + ' <extra_id_1>'
                        self.data[-1]['answer'] = '<extra_id_0> ' + answer
                i += 1
                

                

    def __len__(self):
        return  len(self.data)

    def __getitem__(self,index):
        return self.data[index]

    def collate_fn(self,batch):
        source = self.tokenizer([data['query'] for data in batch], return_tensors='pt', padding=True)
        target = self.tokenizer([data['answer'] for data in batch], return_tensors='pt', padding=True) if not self.test else None


        # Convert inputs to PyTorch tensors
        source_tokens = source['input_ids']
        source_masks = source['attention_mask']

        target_tokens = target['input_ids'] if not self.test else None
        target_masks = target['attention_mask'] if not self.test else None

        tensor_data = {
            "source_tokens": source_tokens,
            "source_masks": source_masks,
            "target_tokens": target_tokens,
            "target_masks": target_masks
        }


        return {"meta_data":batch, "tensor_data":tensor_data}


def model_eval(model, data_loader, tokenizer, beam_size,only_generate=False,save_file=None,multi_gpu=False):
    model.eval()
    loss_toal = 0
    step = 0
    res_temp = []
    res = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader)):

            meta_data = batch['meta_data']
            tensor_data = batch['tensor_data']
            source_tokens = tensor_data['source_tokens'].cuda()
            source_masks = tensor_data['source_masks'].cuda()


            # Generate answer
            max_len = 10
            if multi_gpu:
                generated_ids = model.module.generate(
                    input_ids=source_tokens,
                    attention_mask=source_masks,
                    max_length=max_len,
                    num_beams=beam_size,
                    #do_sample=True,
                    #top_k=50,
                    #top_p=0.95,
                    num_return_sequences=5,
                    #repetition_penalty=2.5
                )
            else:
                generated_ids = model.generate(
                    input_ids=source_tokens,
                    attention_mask=source_masks,
                    max_length=max_len,
                    num_beams=beam_size,
                    # do_sample=True,
                    # top_k=50,
                    # top_p=0.95,
                    num_return_sequences=5,
                    # repetition_penalty=2.5
                )

            #preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in
            #        generated_ids]

            preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


            res_temp.extend(preds)

        for i in range(0, len(res_temp), 5):
            pred_temp = res_temp[i: i + 5]
            res.append(json.dumps(pred_temp))

        if save_file:
            f = open(save_file, 'w', encoding='utf-8')
            csv_writer = csv.writer(f)
            csv_writer.writerow(["id", "ret"])

            for one_data, r in zip(data_loader.dataset.data, res):
                csv_writer.writerow([one_data['qid'], r])
            f.close()

        score = None
        if not only_generate:
            label_df = pd.DataFrame(data_loader.dataset.data)
            label_df['ret'] = res
            label_df.dropna(axis=0, inplace=True)
            score = f1_score(label_df)
            #score = 1

    return score

def model_eval1(model, data_loader, tokenizer, beam_size,only_generate=False,save_file=None,multi_gpu=False):
    model.eval()
    loss_toal = 0
    step = 0
    res_temp = []
    res = []

    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader)):

            meta_data = batch['meta_data']
            tensor_data = batch['tensor_data']
            source_tokens = tensor_data['source_tokens'].cuda()
            source_masks = tensor_data['source_masks'].cuda()


            # Generate answer
            max_len = 10
            if multi_gpu:
                generated_ids = model.module.generate(
                    input_ids=source_tokens,
                    attention_mask=source_masks,
                    max_length=max_len,
                    num_beams=beam_size,
                    #do_sample=True,
                    #top_k=50,
                    #top_p=0.95,
                    num_return_sequences=5,
                    #repetition_penalty=2.5
                )
            else:
                generated_ids = model.generate(
                    input_ids=source_tokens,
                    attention_mask=source_masks,
                    max_length=max_len,
                    num_beams=beam_size,
                    # do_sample=True,
                    # top_k=50,
                    # top_p=0.95,
                    num_return_sequences=5,
                    # repetition_penalty=2.5
                )

            #preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in
            #        generated_ids]

            preds = tokenizer.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)


            res_temp.extend(preds)

        for i in range(0, len(res_temp), 5):
            pred_temp = res_temp[i: i + 5]
            res.append(json.dumps(pred_temp))

        if save_file:
            f = open(save_file, 'w', encoding='utf-8')
            csv_writer = csv.writer(f)
            csv_writer.writerow(["id", "ret"])

            for one_data, r in zip(data_loader.dataset.data, res):
                csv_writer.writerow([one_data['qid'], r])
            f.close()

        score = None
        if not only_generate:
            label_df = pd.DataFrame(data_loader.dataset.data)
            label_df['ret'] = res
            label_df.dropna(axis=0, inplace=True)
            #score = f1_score(label_df)
            score = 1

    return score
