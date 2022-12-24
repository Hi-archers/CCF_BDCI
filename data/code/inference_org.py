import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse

import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import BartForConditionalGeneration, BartTokenizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from utlis import setup_seed, mask_dataset, model_eval1

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging

logging.basicConfig(level=logging.INFO)

setup_seed(42)

def inference(args):

    # Load pre-trained model tokenizer (vocabulary)
    tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_path)

    # Load pre-trained model (weights)
    model = T5ForConditionalGeneration.from_pretrained(args.model_path).cuda()
    model.eval()

    test_dataset = mask_dataset(args.val_path, tokenizer,test=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=args.val_batch_size,  num_workers=3, collate_fn=test_dataset.collate_fn)

    score = model_eval1(model, test_dataloader, tokenizer, args.beam_size, only_generate=args.only_generate,save_file=args.result_file)
    #score = model_eval(model, test_dataloader, tokenizer,10)

    print("score:",score)


def parse_args():
    parser = argparse.ArgumentParser('Testing args...')
    parser.add_argument('--model_path', default="../user_data/model/t5-large-ssm-org/best_ckpt", help='Model path.')
    parser.add_argument('--tokenizer_path', default="../user_data/model/t5-large-ssm",help='Model path.')
    parser.add_argument('--val_batch_size', default=20, help='Validation set batch size.')
    parser.add_argument('--val_path', default='../raw_data/KCT_test_public.txt', help='Validation set file.')
    parser.add_argument('--only_generate', default=False, help='Whether validate or not.')
    parser.add_argument('--result_file', default='../user_data/result/org.csv', help='Save result path.')
    parser.add_argument('--beam_size', default=10, help='Size of beam search.')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print(args)

    inference(args)
    a = 1