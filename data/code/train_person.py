import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
path=os.path.dirname(os.path.abspath(__file__))  
sys.path.append(path) 
print("测试path: ",path)
import test
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import BartForConditionalGeneration, BartTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

import argparse

from utlis import setup_seed, mask_dataset, model_eval

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging

logging.basicConfig(level=logging.INFO)

setup_seed(42)

def train(args):
    model_name = args.model_name
    train_batch_size = args.train_batch_size
    val_batch_size = args.val_batch_size
    train_path = args.train_path
    val_path = args.val_path
    save_file = args.save_file
    epoch_number = args.epoch_number
    learning_rate = args.learning_rate
    beam_size = args.beam_size
    multi_gpu = args.multi_gpu

    writer_train = SummaryWriter(save_file + '/log/train')
    writer_val = SummaryWriter(save_file + '/log/val')

    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).cuda()
    model.train()

    if multi_gpu:
        model = torch.nn.DataParallel(model)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    train_dataset = mask_dataset(train_path, tokenizer)
    val_dataset = mask_dataset(val_path, tokenizer,test=True )

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=3, collate_fn=train_dataset.collate_fn)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=val_batch_size, num_workers=3, collate_fn=val_dataset.collate_fn)

    best_score = -1
    best_epoch = 0

    print("Start training...")
    for epoch in range(epoch_number):
        loss_epoch = 0
        step = 0
        print("Epoch",epoch)
    #    score = model_eval(model, val_dataloader, tokenizer, model_type)
        for i,batch in enumerate(tqdm(train_dataloader)):
            model.train()
            meta_data = batch['meta_data']
            tensor_data = batch['tensor_data']
            source_tokens = tensor_data['source_tokens'].cuda()
            source_masks = tensor_data['source_masks'].cuda()
            target_tokens = tensor_data['target_tokens'].cuda()
            target_masks = tensor_data['target_masks'].cuda()

            pad_token_id = tokenizer.pad_token_id
            target_tokens[target_tokens == pad_token_id] = -100
            optimizer.zero_grad()

            # Run forward pass and calculate loss
            outputs = model(source_tokens, attention_mask=source_masks, decoder_attention_mask =None, labels=target_tokens)
            loss = outputs.loss
            if multi_gpu:
                loss = loss.mean()
            loss_epoch += loss.item()
            loss.backward()
            optimizer.step()

            step += 1
        loss_train = loss_epoch / step
        score = model_eval(model, val_dataloader, tokenizer,beam_size,False,None,multi_gpu)
        print("Val score", score)

        if multi_gpu:
            model.module.save_pretrained(os.path.join(save_file, 'last_ckpt'))
        else:
            model.save_pretrained(os.path.join(save_file, 'last_ckpt'))

        writer_train.add_scalar('loss', loss_train, step)
        writer_val.add_scalar('score', score, step)

        if score > best_score:
            best_score = score
            best_epoch = epoch
            print('New best score', best_score)
            if multi_gpu:
                model.module.save_pretrained( os.path.join(save_file, 'best_ckpt'))
            else:
                model.save_pretrained(os.path.join(save_file, 'best_ckpt'))

    print('Finish training...')
    print('Best epoch:', best_epoch)
    print('Best score:', best_score)

def parse_args():
    parser = argparse.ArgumentParser('Training args...')
    parser.add_argument('--model_name', default='../user_data/model/t5-large-ssm', help='Model name.')
    # parser.add_argument('--model_name', default='/ssd2/fanxiaoran/workspace/CCIR_CUP_2021/data/user_data/model/t5-large-ssm', help='Model name.')

    parser.add_argument('--train_batch_size', default=20, help='Traning set batch size.')
    parser.add_argument('--val_batch_size', default=20, help='Validation set batch size.')
    parser.add_argument('--train_path', default='../user_data/train_data/person_train.txt', help='traning set file.')
    parser.add_argument('--val_path', default=  '../user_data/train_data/person_val.txt', help='validation set file.')
    parser.add_argument('--save_file', default= '../user_data/model/t5-large-ssm-person', help='save model path.')
    parser.add_argument('--epoch_number', default=40, help='Epoch number.')
    parser.add_argument('--learning_rate', default=0.00005, help='Learning rate.')
    parser.add_argument('--beam_size', default=10, help='Size of beam search.')
    parser.add_argument('--multi_gpu', default=False, help='Multi-GPU.')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print(args)

    train(args)
    a = 1
