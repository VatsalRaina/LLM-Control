import argparse
import os
import sys
import json
import time
import openai
from unidecode import unidecode
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--test_path', type=str, help='Load path of test data')
parser.add_argument('--save_path', type=str, help='Load path to save results with paraphrases')

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
device = get_device()
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
model.eval().to(device)

def get_device():
    if torch.cuda.is_available():

        print('WE HAVE CUDA')
        return torch.device('cuda')
    else:
         return torch.device('cpu')

def generate_paraphrase(data, column_name):

    prompt = "Paraphrase this document with a negative statement \n"
    input_ids = tokenizer(data[column_name], return_tensors="pt").input_ids
    outputs = model.generate(input_ids.to(device),max_length=300, do_sample=True)
   
    par_doc = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return {'par_document' : par_doc}

def main(args):
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')




    with open(args.test_path, 'r', encoding='utf-8') as f:
        test_phrases = [unidecode(line.strip().replace('@@ ', '').replace('\t', ' ')) for line in f.readlines()]

    dataset = Dataset.from_dict({'document' : test_phrases, })

    # dataset = dataset.select(range(100))
    dataset = dataset.map(generate_paraphrase, batched=False, fn_kwargs={"column_name" : 'document'})
    print(dataset)