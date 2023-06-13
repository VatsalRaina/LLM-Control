#! /usr/bin/env python

import argparse
import os
import sys
import json
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


    openai.api_key = args.api_key

    with open(args.save_path) as f:
        expanded_examples = json.load(f)

    start_point = len(expanded_examples)

    with open(args.subset_path) as f:
        sample_data = json.load(f)

    batch_examples = []
    for count, item in enumerate(sample_data):
        if count < start_point: continue
        print(count)
        context, question, answer, distractors = item['context'], item['question'], item['answer'], item['distractors']
        distractors_string = ""
        for dist in distractors:
            distractors_string += dist + ' ; '
        distractors_string = distractors_string[:-3]
        prompt = "Consider a multiple-choice question with the following context:\n" + context + "\nThe question is:\n" + question + "\nThe correct answer is:\n" + answer + "\nThe distractor options are:\n" + distractors_string + "\nCan you make the distractors less plausible.\nPlease return only the new distractors with each one separated by a ; and no explanations."
        # print(prompt,'\n')
        model = "gpt-3.5-turbo"
        response = openai.ChatCompletion.create(model=model, messages=[{"role": "user", "content": prompt}])
        generated_text = response.choices[0].message.content.strip()
        distractors_more_plausible = generated_text.split(';')
        curr_example = {'context': context, 'question': question, 'answer': answer, 'distractors': distractors, 'distractors_more_plausible': distractors_more_plausible}
        batch_examples.append(curr_example)
        if len(batch_examples) == 2:
            expanded_examples += batch_examples
            batch_examples = []
            with open(args.save_path, 'w') as f:
                json.dump(expanded_examples, f)
            print("Saved up to:", count)
            print("----------------------")


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
