#! /usr/bin/env python

from datasets import load_dataset
import argparse
import os
import sys
import json
import time
import openai

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--test_path', type=str, help='Load path of test data')
parser.add_argument('--api_key', type=str, help='Load OpenAI API key')
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

    dataset = load_dataset("squad_v2")['validation']

    contexts = []
    for ex in dataset:
        ctxt = ex["context"]
        if ctxt not in contexts:
            contexts.append(ctxt)

    batch_examples = []
    for count, item in enumerate(contexts):
        if count < start_point: continue
        print(count)
        # get paraphrase here
        prompt = item + "\nParaphrase this document"
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