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
# model_name_or_path = "/ssd2/fanxiaoran/workspace/CCIR_CUP_2021/data/user_data/model/bert-base-uncased"
# 同时试试roberta-base-chinese

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


def train(model, dataloader, optimizer_, scheduler_, device_):

    # Tracking variables.
    predictions_labels = []
    true_labels = []
    # Total loss for this epoch.
    total_loss = 0

    # Put the model into training mode.
    model.train()

    # For each batch of training data...
    for batch in tqdm(dataloader, total=len(dataloader)):

        # Add original labels - use later for evaluation.
        true_labels += batch['labels'].numpy().flatten().tolist()

        # move batch to device
        batch = {k: v.type(torch.long).to(device_) for k, v in batch.items()}

        # Always clear any previously calculated gradients before performing a
        # backward pass.
        model.zero_grad()

        # Perform a forward pass (evaluate the model on this training batch).
        # This will return the loss (rather than the model output) because we
        # have provided the `labels`.
        # The documentation for this a bert model function is here:
        # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
        outputs = model(**batch)

        # The call to `model` always returns a tuple, so we need to pull the
        # loss value out of the tuple along with the logits. We will use logits
        # later to calculate training accuracy.
        loss, logits = outputs[:2]

        # Accumulate the training loss over all of the batches so that we can
        # calculate the average loss at the end. `loss` is a Tensor containing a
        # single value; the `.item()` function just returns the Python value
        # from the tensor.
        total_loss += loss.item()

        # Perform a backward pass to calculate the gradients.
        loss.backward()

        # Clip the norm of the gradients to 1.0.
        # This is to help prevent the "exploding gradients" problem.
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Update parameters and take a step using the computed gradient.
        # The optimizer dictates the "update rule"--how the parameters are
        # modified based on their gradients, the learning rate, etc.
        optimizer.step()

        # Update the learning rate.
        scheduler.step()

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()

        # Convert these logits to list of predicted labels values.
        predictions_labels += logits.argmax(axis=-1).flatten().tolist()

    # Calculate the average loss over the training data.
    avg_epoch_loss = total_loss / len(dataloader)

    # Return all true labels and prediction for future evaluations.
    return true_labels, predictions_labels, avg_epoch_loss


def validation(model, dataloader, device_):


    # print("进来了这个函数...")
    # Tracking variables
    predictions_labels = []
    true_labels = []
    # total loss for this epoch.
    total_loss = 0

    # Put the model in evaluation mode--the dropout layers behave differently
    # during evaluation.
    model.eval()

    # Evaluate data for one epoch
    for batch in tqdm(dataloader, total=len(dataloader)):

        # add original labels
        true_labels += batch['labels'].numpy().flatten().tolist()

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

            # The call to `model` always returns a tuple, so we need to pull the
            # loss value out of the tuple along with the logits. We will use logits
            # later to to calculate training accuracy.
            loss, logits = outputs[:2]

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
            total_loss += loss.item()

            # update list
            predictions_labels += logits.argmax(axis=-1).flatten().tolist()

    # Calculate the average loss over the training data.
    avg_epoch_loss = total_loss / len(dataloader)

    # Return all true labels and prediciton for future evaluations.
    return true_labels, predictions_labels, avg_epoch_loss


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
model_config = AutoConfig.from_pretrained(pretrained_model_name_or_path=model_name_or_path,
                                          num_labels=n_labels)

# Get model's tokenizer.
print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name_or_path)

# Get the actual model.
print('Loading model...')
model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_name_or_path,
                                                           config=model_config)

# Load model to defined device.
model.to(device)
print('Model loaded to `%s`' % device)


print('Dealing with Train Dataset...')
# Create pytorch dataset.

train_dataset = TextDataset(data=pd.read_csv('../user_data/train_data/bert_train.csv',encoding = 'utf-8'),
                            use_tokenizer=tokenizer,
                            labels_ids=labels_ids,
                            max_sequence_len=max_length)
print('Created `train_dataset` with %d examples!' % len(train_dataset))

# Move pytorch dataset into dataloader.
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)
print('Created `train_dataloader` with %d batches!' % len(train_dataloader))

print('Dealing with Valid Dataset...')
# Create pytorch dataset.

valid_dataset = TextDataset(data=pd.read_csv('../user_data/train_data/bert_val.csv',encoding = 'utf-8'),
                            use_tokenizer=tokenizer,
                            labels_ids=labels_ids,
                            max_sequence_len=max_length)
print('Created `valid_dataset` with %d examples!' % len(valid_dataset))


# Move pytorch dataset into dataloader.
valid_dataloader = DataLoader(
    valid_dataset, batch_size=batch_size, shuffle=False)
print('Created `eval_dataloader` with %d batches!' % len(valid_dataloader))



# Note: AdamW is a class from the huggingface library (as opposed to pytorch)
# I believe the 'W' stands for 'Weight Decay fix"
optimizer = AdamW(model.parameters(),
                  lr=3e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                  eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                  )

# Total number of training steps is number of batches * number of epochs.
# `train_dataloader` contains batched data so `len(train_dataloader)` gives
# us the number of batches.
total_steps = len(train_dataloader) * epochs

# Create the learning rate scheduler.
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=200,  # Default value 0 in run_glue.py
                                            num_training_steps=total_steps)

# Store the average loss after each epoch so we can plot them.
all_loss = {'train_loss': [], 'val_loss': []}
all_acc = {'train_acc': [], 'val_acc': []}

best_acc = 0.0

# Loop through each epoch.
print('Epoch')
for epoch in tqdm(range(epochs)):
    print()
    print('Training on batches...')
    # Perform one full pass over the training set.
    train_labels, train_predict, train_loss = train(
        model, train_dataloader, optimizer, scheduler, device)
    train_acc = accuracy_score(train_labels, train_predict)

    # Get prediction form model on validation data.
    print('Validation on batches...')
    valid_labels, valid_predict, val_loss = validation(
        model, valid_dataloader, device)
    val_acc = accuracy_score(valid_labels, valid_predict)

    # Print loss and accuracy values to see how training evolves.
    print("  train_loss: %.5f - val_loss: %.5f - train_acc: %.5f - valid_acc: %.5f\n" %
          (train_loss, val_loss, train_acc, val_acc))

    if val_acc > best_acc:
        best_acc = val_acc
        best_epoch = epoch + 1
        # Save model checkpoint
        output_dir = "../user_data/model/best_checkpoint"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        model.save_pretrained(output_dir)
        tokenizer.save_vocabulary(output_dir)

        print("Save the best model!")

    # Store the loss value for plotting the learning curve.
    all_loss['train_loss'].append(train_loss)
    all_loss['val_loss'].append(val_loss)
    all_acc['train_acc'].append(train_acc)
    all_acc['val_acc'].append(val_acc)



print('Dealing with Test Dataset...')
# Create pytorch dataset.
test_dataset = TextDataset(
    data=pd.read_csv('../user_data/train_data/bert_test.csv',encoding = 'utf-8'),
    use_tokenizer=tokenizer,
    labels_ids=labels_ids,
    max_sequence_len=max_length
)
print('Created `test_dataset` with %d examples!' % len(test_dataset))

# Move pytorch dataset into dataloader.
test_dataloader = DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False)
print('Created `test_dataloader` with %d batches!' % len(test_dataloader))

# Get prediction form model on validation data. This is where you should use
# your test data.
true_labels, predictions_labels, avg_epoch_loss = validation(model,
                                                             test_dataloader, device)


# print("debuging true_labels, predictions_labels: ",true_labels, predictions_labels)
# Create the evaluation report.
evaluation_report = classification_report(
    true_labels, predictions_labels, labels=list(labels_ids))


# 生成测试集的部分
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
print('bert is finished...')

