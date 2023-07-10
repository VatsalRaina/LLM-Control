#! /usr/bin/env python

import argparse
import os
import sys
import json
from evaluate import load
from jiwer import wer
import numpy as np

from transformers import pipeline
classification = pipeline('sentiment-analysis')

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--candidate_path', type=str, help='Load path of test data')

def get_sentiment(predictions):
    label_and_scores = classification(predictions)
    scores = [ex['score'] for ex in label_and_scores]
    return scores

def get_bert_score(predictions, references):

    bertscore = load("bertscore")
    results = bertscore.compute(predictions=predictions, references=references, model_type="distilbert-base-uncased")
    return results

def get_wer(predictions, references):

    wer_scores = []

    for pred, ref in zip(predictions, references):
        wer_scores.append(wer(ref, pred))

    return wer_scores


def main(args):
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')


    with open(args.candidate_path) as f:
        examples = json.load(f)

    references = [ex['original'] for ex in examples]
    predictions = [ex['paraphrase'] for ex in examples]

    sentiment_score = get_sentiment(predictions, references)
    bert_score = get_bert_score(predictions, references)
    wer_score = get_wer(predictions, references)

    print('sentiment_mean : ', np.round(np.mean(sentiment_score)* 100, 1), 'sentiment_std : ', np.round(np.std(sentiment_score)* 100, 1))

    print('f1_mean : ', np.round(np.mean(bert_score['f1']) * 100, 1), 'f1_std : ', np.round(np.std(bert_score['f1']) * 100, 1))
    print('recall_mean : ',np.round( np.mean(bert_score['recall'])* 100, 1), 'recall_std : ', np.round(np.std(bert_score['recall'] )* 100, 1))
    print('precision_mean : ', np.round(np.mean(bert_score['precision'])* 100, 1), 'precision_std : ', np.round(np.std(bert_score['precision'])* 100, 1))

    print('wer_mean : ', np.round(np.mean(wer_score)* 100, 1), 'wer_std : ', np.round(np.std(wer_score)* 100, 1))




if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

