#! /usr/bin/env python

import argparse
import os
import sys
import json
import time
import openai
from unidecode import unidecode

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--test_path', type=str, help='Load path of test data')
parser.add_argument('--api_key', type=str, help='Load OpenAI API key')
parser.add_argument('--target_sentiment', type=str, help='Set the target sentiment of the paraphrase')
parser.add_argument('--save_path', type=str, help='Load path to save results with paraphrases')


def main(args):
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')


    openai.api_key = args.api_key

    with open(args.save_path) as f:
        expanded_examples = json.load(f)

    start_point = len(expanded_examples)

    with open(args.test_path, 'r', encoding='utf-8') as f:
        test_phrases = [unidecode(line.strip().replace('@@ ', '').replace('\t', ' ')) for line in f.readlines()]

    batch_examples = []
    for count, item in enumerate(test_phrases):
        if count < start_point: continue
        print(count)
        # get paraphrase here
        prompt = item + "\nIf sentiment is a score from 0 to 10 where 0 is negative and 10 is positive, paraphrase this document with a sentiment score of " + args.target_sentiment
        # print(prompt,'\n')
        model = "gpt-3.5-turbo"
        response = openai.ChatCompletion.create(model=model, messages=[{"role": "user", "content": prompt}])
        generated_text = response.choices[0].message.content.strip()
        curr_example = {'original':item, 'paraphrase':generated_text}
        batch_examples.append(curr_example)
        if len(batch_examples) == 1:
            expanded_examples += batch_examples
            batch_examples = []
            with open(args.save_path, 'w') as f:
                json.dump(expanded_examples, f)
            print("Saved up to:", count)
            print("----------------------")


if __name__ == '__main__':
    args = parser.parse_args()

    for count in range(1,100):
        try:
            main(args)
            time.sleep(1)
        except openai.error.RateLimitError:
            print("openai.error.RateLimitError... #{}".format(count))
            print("restart in 10 seconds")
            time.sleep(10)
        except openai.error.ServiceUnavailableError:
            print("openai.error.ServiceUnavailableError... #{}".format(count))
            print("restart in 10 seconds")
            time.sleep(10)
        except openai.error.APIError:
            print("openai.error.APIError... #{}".format(count))
            print("restart in 20 seconds")
            time.sleep(20)