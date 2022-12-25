import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import re
import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, accuracy_score
from transformers import (AutoConfig,
                          AutoModelForSequenceClassification,
                          AutoTokenizer, AdamW,
                          get_linear_schedule_with_warmup,
                          set_seed,
                          )


# Set seed for reproducibility,
set_seed(42)

# Number of training epochs (authors recommend between 2 and 4)

epochs = 10

# Number of batch_size - depending on the max sequence length and GPU memory.
# For 512 sequence length batch of 10 works without cuda memory issues.
# For small sequence length can try batch of 32 or higher.
batch_size = 64

# Padd or truncate text sequences to a specific length
# if `None` it will use maximum sequence of word piece tokens allowed by model.
max_length = 256

# Look for gpu to use. Will use `cpu` by default if no gpu found.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("current_device: ",device)
# Name of transformers model - will use already pretrained model.
# Path of transformer model - will load your own model from local disk.


model_name_or_path = '../user_data/model/bert-base-uncased'

# Dicitonary of labels and their id - this will be used to convert.
# String labels to number.
# 加上了3个标签
labels_ids = [0, 1, 2, 3, 4, 5]


# How many labels are we using in training.
# This is used to decide size of classification head.
n_labels = len(labels_ids)

class TextDataset(Dataset):

    def __init__(self, data, use_tokenizer, labels_ids, max_sequence_len=None, mode=''):
        # Check max sequence length.
        max_sequence_len = use_tokenizer.max_len if max_sequence_len is None else max_sequence_len
        texts = []
        labels = []
        print('Reading partitions...')
        texts = list(data.text)
        if mode != 'predict':
            labels = list(data.labels)

        else:
            print("这里处理真正的public_test的数据集...")
        # Number of exmaples.
        self.n_examples = len(texts)
        # Use tokenizer on texts. This can take a while.
        print('Using tokenizer on all texts. This can take a while...')
        # self.inputs = use_tokenizer(texts, add_special_tokens=True, truncation=True,
        #                             padding=True, return_tensors='pt',  max_length=max_sequence_len)

        print("debuging max_seq_len: ",max_sequence_len)
        # print("texts: ",texts)
        self.inputs = use_tokenizer(texts, add_special_tokens=True, truncation=True,
                                    padding=True, return_tensors='pt',  max_length=max_sequence_len)
        print("not run in this place...")
        # Get maximum sequence length.
        self.sequence_len = self.inputs['input_ids'].shape[-1]
        print('Texts padded or truncated to %d length!' % self.sequence_len)
        # Add labels.
        if mode != 'predict':
            self.inputs.update({'labels': torch.tensor(labels)})
        print('Finished!\n')

        return

    def __len__(self):
        return self.n_examples

    def __getitem__(self, item):
        return {key: self.inputs[key][item] for key in self.inputs.keys()}



def prediction(model, dataloader, device_):

    # Tracking variables
    predictions_labels = []

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Evaluate data for one epoch
    for batch in tqdm(dataloader, total=len(dataloader)):

        # move batch to device
        batch = {k: v.type(torch.long).to(device_) for k, v in batch.items()}

        # Telling the model not to compute or store gradients, saving memory and
        # speeding up validation
        with torch.no_grad():

            # Forward pass, calculate logit predictions.
            # This will return the logits rather than the loss because we have
            # not provided labels.
            # token_type_ids is the same as the "segment ids", which
            # differentiates sentence 1 and 2 in 2-sentence tasks.
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            outputs = model(**batch)

            # print("outputs: ",outputs)
            # The call to `model` always returns a tuple, so we need to pull the
            # loss value out of the tuple along with the logits. We will use logits
            # later to to calculate training accuracy.
            # pre
            # loss, logits = outputs[:2]

            logits = outputs['logits']

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()

            # update list
            predictions_labels += logits.argmax(axis=-1).flatten().tolist()

    # Return prediciton labels.
    return predictions_labels

# Get model configuration.
print('Loading configuraiton...')
model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path='../user_data/model/best_checkpoint',
                                          num_labels=n_labels)

# Get model's tokenizer.
print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path='../user_data/model/best_checkpoint')

# Get the actual model.
print('Loading model...')
model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path='../user_data/model/best_checkpoint',
                                                           config=model_config)

# Load model to defined device.
model.to(device)
print('Model loaded to `%s`' % device)



# 读取的该temp_test只是拿到其文本,放到模型预测
sequences = pd.read_csv('../user_data/train_data/no_use_temp_test_1.csv',encoding = 'utf-8')

test_dataset = TextDataset(data=sequences,
                           use_tokenizer=tokenizer,
                           labels_ids=labels_ids,
                           max_sequence_len=max_length,
                           mode='predict')
print('Created `test_dataset` with %d examples!' % len(test_dataset))

# Move pytorch dataset into dataloader.
test_dataloader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False)
print('Created `test_dataloader` with %d batches!' % len(test_dataloader))

# Get prediction form model on prediction data. This is where you should use
# your test data.
prediction_labels = prediction(model, test_dataloader, device)

sequences['predict'] = prediction_labels
#print("最终预测的test_set的标签为: ",len(prediction_labels),prediction_labels)
sequences = sequences[['labels', 'predict', 'text']]
# print(sequences)

qids = []
label = []
text = []


cnt = 0
# 该路径只是拿到test的qid用于后续T5模型对于test数据的concat处理
test_data_path = '../user_data/train_data/no_use_temp_test_2.csv'
gen_test_label_path = '../user_data/result/gen_test_lb.csv'
with open(test_data_path, encoding = 'utf-8') as f:
    Dict = pd.read_csv(f, encoding='utf-8')
    Dict['label'] = -1
    for i in range(len(Dict)):
        one_data = Dict.loc[i]
        one_data['label'] = prediction_labels[i]
        # print(i,one_data['label'])

        qids.append(one_data['qid'])
        text.append(one_data['query'])
        label.append(one_data['label'])

gen_test_dict = pd.DataFrame({
    'qids':qids,
    'query':text,
    'label':label,
})

gen_test_dict.to_csv(gen_test_label_path,index = False,encoding = 'utf-8')
print('bert inference is finished...')

