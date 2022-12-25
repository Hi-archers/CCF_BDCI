import flask
from flask import Flask, request, render_template
import json
import numpy as np
import torch
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from transformers import BertTokenizer, BartForConditionalGeneration
import re
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

from transformers import BertTokenizer, BertForSequenceClassification

def Load_BERT():
    tokenizer = BertTokenizer.from_pretrained("../../model/bert-base-uncased/")
    model = BertForSequenceClassification.from_pretrained("../data/bert-base/best_ckpt/").cuda()

    #model = BertForSequenceClassification.from_pretrained("../../model/bert-base-uncased/").cuda()
    model.eval()
    return model, tokenizer

from transformers import T5Tokenizer, T5ForConditionalGeneration

def Load_T5(model_path):
    model = T5ForConditionalGeneration.from_pretrained(model_path).cuda() #other
    tokenize = T5Tokenizer.from_pretrained("../../model/t5-large-ssm/")

    model.eval()
    return model, tokenize

app = Flask(__name__)

def _generate(sentence, num_sentences, beam_search):
    #对输入内容进行处理
    print(sentence)

    BERT_input = sentence.replace('[MASK]',"")
    T5_input = sentence.replace('[MASK]',"<extra_id_0>")

    BERT, BERT_tokenizer = Load_BERT()

    print(BERT_input)

    BERT_input = BERT_tokenizer.encode(BERT_input, return_tensors="pt").cuda()

    output = BERT(input_ids=BERT_input)['logits']

    print(output)

    output = torch.argmax(output, dim=1)
    output = output.item()

    print(output)

    if output == 0:
        model,tokenize = Load_T5("../data/t5-large-ssm-batch32-new/best_ckpt/")
        input = tokenize.encode(T5_input, return_tensors="pt").cuda()

        print(f"Load model0")

        predictions = model.generate(
            input_ids = input,
            num_beams = beam_search,
            num_return_sequences = num_sentences,
        )

        final_outputs = tokenize.batch_decode(predictions, skip_special_tokens=True)

    if output == 1:
        model0,tokenize = Load_T5("../data/t5-large-ssm-year/best_ckpt/")
        model1,_        = Load_T5("../data/t5-large-ssm-mon/best_ckpt/")
        model2,_        = Load_T5("../data/t5-large-ssm-day/best_ckpt/")

        print(f"Load model1")

        input = tokenize.encode(T5_input, return_tensors="pt").cuda()

        predictions0 = model0.generate(
            input_ids = input,
            max_new_tokens=3,
            num_beams = beam_search,
            num_return_sequences = num_sentences,
        )

        predictions1 = model1.generate(
            input_ids = input,
            max_new_tokens=3,
            num_beams = beam_search,
            num_return_sequences = num_sentences,
        )

        predictions2 = model2.generate(
            input_ids=input,
            max_new_tokens=3,
            num_beams = beam_search,
            num_return_sequences = num_sentences,
        )

        final_outputs = []
        tmp0 = tokenize.batch_decode(predictions0, skip_special_tokens=True)
        tmp1 = tokenize.batch_decode(predictions1, skip_special_tokens=True)
        tmp2 = tokenize.batch_decode(predictions2, skip_special_tokens=True)

        for k in range(num_sentences):
            final_outputs.append(tmp0[k] + ' ' + tmp1[k] + ' ' + tmp2[k])

    if output == 2:
        model,tokenize = Load_T5("../data/t5-large-ssm-yearday/best_ckpt/")
        input = tokenize.encode(T5_input, return_tensors="pt").cuda()

        print(f"Load model2")
        predictions = model.generate(
            input_ids = input,
            num_beams = beam_search,
            num_return_sequences = num_sentences,
        )
        final_outputs = tokenize.batch_decode(predictions, skip_special_tokens=True)

    if output == 3:
        model,tokenize = Load_T5("../data/t5-large-ssm-local/best_ckpt/")
        input = tokenize.encode(T5_input, return_tensors="pt").cuda()

        print(f"Load model3")
        predictions = model.generate(
            input_ids = input,
            num_beams = beam_search,
            num_return_sequences = num_sentences,
        )
        final_outputs = tokenize.batch_decode(predictions, skip_special_tokens=True)

    if output == 4:
        model,tokenize = Load_T5("../data/t5-large-ssm-person/best_ckpt/")
        input = tokenize.encode(T5_input, return_tensors="pt").cuda()

        print(f"Load model4")
        predictions = model.generate(
            input_ids = input,
            num_beams = beam_search,
            num_return_sequences = num_sentences,
        )
        final_outputs = tokenize.batch_decode(predictions, skip_special_tokens=True)


    if output == 5:
        model,tokenize = Load_T5("../data/t5-large-ssm-org/best_ckpt/")
        input = tokenize.encode(T5_input, return_tensors="pt").cuda()

        print(f"Load model5")
        predictions = model.generate(
            input_ids = input,
            num_beams = beam_search,
            num_return_sequences = num_sentences,
        )
        final_outputs = tokenize.batch_decode(predictions, skip_special_tokens=True)

    return output, final_outputs


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_paraphrase', methods=['post'])
def get_paraphrase():
    try:
        input_text = request.json['input_text']
        num_sentences = int(request.json['num_sentences'])
        beam_search = int(request.json['beam_search'])

        label, response = _generate(input_text, num_sentences, beam_search)

        question = [
            "This is a Other Problem",
            "This is a Date Problem",
            "This is a Year Problem",
            "This is a local Problem",
            "This is a Person Problem",
            "This is a Organization Problem",
        ]

        response.insert(0,"")
        response.insert(0,question[label])

        str_response = '\n'.join([r for r in response])

        return app.response_class(response=json.dumps(str_response), status=200, mimetype='application/json')
    except Exception as error:
        err = str(error)
        print(err)
        return app.response_class(response=json.dumps(err), status=500, mimetype='application/json')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=65428, use_reloader=True)
