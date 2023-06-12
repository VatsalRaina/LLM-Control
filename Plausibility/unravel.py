#! /usr/bin/env python

import argparse
import os
import sys
import json
import numpy as np
import random

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--test_data_path', type=str, help='Load path of test data')
parser.add_argument('--save_path', type=str, help='Load path of test data')


def main(args):
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')


    with open(args.test_data_path + "middle.json") as f:
        middle_data = json.load(f)
    with open(args.test_data_path + "high.json") as f:
        high_data = json.load(f)
    with open(args.test_data_path + "college.json") as f:
        college_data = json.load(f)
    test_data = middle_data + high_data + college_data

    def asNum(x):
        if x=="A":
            return 0
        if x=="B":
            return 1
        if x=="C":
            return 2
        if x=="D":
            return 3

    # Unwrap the questions into single examples
    all_examples = []
    for item in test_data:
        context = item["article"]
        questions = item["questions"]
        options = item["options"]
        answers = item["answers"]
        for qu_num, question in enumerate(questions):
            lab = asNum(answers[qu_num])
            opts = options[qu_num]
            answer = opts[lab]
            distractors = [x for x_count, x in enumerate(opts) if x_count!=lab]
            curr_example = {'context': context, 'question': question, 'answer': answer, 'distractors': distractors}
            all_examples.append(curr_example)

    with open(args.save_path, 'w') as f:
        json.dump(all_examples, f)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
