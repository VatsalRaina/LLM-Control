#! /usr/bin/env python

import argparse
import os
import sys
import json
import numpy as np
import random
import openai

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--subset_path', type=str, help='Load path of test data')
parser.add_argument('--api_key', type=str, help='Load OpenAI API key')
parser.add_argument('--save_path', type=str, help='Load path to save results with increased distractors')


def main(args):
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)

    openai.api_key = args.api_key

    with open(args.subset_path) as f:
        sample_data = json.load(f)

    expanded_examples = []
    for count, item in enumerate(sample_data):
        print(count)
        context, question, answer, distractors = item['context'], item['question'], item['answer'], item['distractors']
        distractors_string = ""
        for dist in distractors:
            distractors_string += dist + ' ; '
        distractors_string = distractors_string[:-3]
        prompt = "Consider a multiple-choice question with the following context:\n" + context + "\nThe question is:\n" + question + "\nThe correct answer is:\n" + answer + "\nThe distractor options are:\n" + distractors_string + "\nCan you make the distractors more plausible.\nPlease return only the new distractors with each one separated by a ; and no explanations."
        print(prompt,'\n')
        model = "gpt-3.5-turbo"
        response = openai.Completion.create(engine=model, prompt=prompt, max_tokens=200)
        generated_text = response.choices[0].text.strip()
        distractors_more_plausible = generated_text.split(';')
        curr_example = {'context': context, 'question': question, 'answer': answer, 'distractors': distractors, 'distractors_more_plausible': distractors_more_plausible}
        expanded_examples.append(curr_example)
        break


    with open(args.save_path, 'w') as f:
        json.dump(expanded_examples, f)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
