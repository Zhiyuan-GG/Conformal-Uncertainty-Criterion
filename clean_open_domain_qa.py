import argparse
import pathlib
import pickle
import tqdm
import os
import random
import json

import accelerate
import datasets
from transformers import AutoModelForCausalLM, AutoTokenizer

import numpy as np
import torch


parser = argparse.ArgumentParser()
parser.add_argument('--data-dir',
                    default='../datasets',
                    help='save parsed dataset')
parser.add_argument('--cache-dir',
                    default='../cache',
                    help='cache model from hugging face')
parser.add_argument('--fraction_of_data_to_use', type=float, default=1.0)
# ----------------------------------------------------------------------------------------------------------------------
# for run name
parser.add_argument('--record-dir',
                    default='../records',
                    help='save experimental records')
parser.add_argument('--generate-model',
                    default=r'D:\projects\cache_model\Qwen2.5-14B-Instruct',
                    help='local path of generative llm')
parser.add_argument('--dataset', default='coqa')
parser.add_argument('--max-length-of-generation', type=int, default=128)
parser.add_argument('--sample', type=bool, default=True, help='sampled or the most likely')  ##
parser.add_argument('--num-beams', type=int, default=5, help='for the most likely generation')
parser.add_argument('--num-generations-per-prompt', type=int, default=10, help='for sampling')
parser.add_argument('--top-p', type=float, default=0.9, help='for sampling')
parser.add_argument('--temperature', type=float, default=1.0, help='for sampling')
# ----------------------------------------------------------------------------------------------------------------------
args = parser.parse_args()
# run_name for saving experimental record
model_name = args.generate_model.split('\\')[-1] if '\\' in args.generate_model else args.generate_model
if args.dataset in ['coqa', 'triviaqa']:
    args.max_length_of_generation = 36
if args.sample:
    run_name = os.path.join(args.record_dir,
                            args.dataset,
                            model_name,
                            'num_generations-' + str(args.num_generations_per_prompt),
                            'temperature-' + str(args.temperature),
                            'max_len_of_generation-' + str(args.max_length_of_generation))
else:
    run_name = os.path.join(args.record_dir,
                            args.dataset,
                            model_name,
                            'num_beams-' + str(args.num_beams),
                            'max_len_of_generation-' + str(args.max_length_of_generation))
# ----------------------------------------------------------------------------------------------------------------------
# Set seed for recurrence
seed_value = 10
# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
os.environ['PYTHONHASHSEED'] = str(seed_value)
# 2. Set `python` built-in pseudo-random generator at a fixed value
random.seed(seed_value)
# 3. Set `numpy` pseudo-random generator at a fixed value
np.random.seed(seed_value)
# 4. Fix torch random seed
torch.manual_seed(seed_value)
# cache path for hf_datasets
os.environ["HF_DATASETS_CACHE"] = args.cache_dir
# set cuda device 0,1
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
# ----------------------------------------------------------------------------------------------------------------------
# load generation
with open(f'{run_name}/generations.pkl', 'rb') as record_file:
    generations = pickle.load(record_file)

# clean function
def filter(text):
    text = text.strip()
    strings_to_filter_on = ['\n', '.', '#']
    # for decimal point
    while '.' in text:
        point_idx = text.index('.')  # current first point
        if point_idx != len(text) - 1 and text[point_idx - 1].isdigit() and text[point_idx + 1].isdigit():
            text = text.replace('.', '(dp)', 1)
        else:  # 当前第一个point不是小数点，即句号，跳出循环，开始filter
            break
    for string in strings_to_filter_on:
        if string in text:
            text = text.split(string)[0]
    if '(dp)' in text:
        text = text.replace('(dp)', '.')
    new_text = ''
    if len(text) != 0:
        # english
        for ch in text:
            if ch.isascii():
                new_text += ch
    if len(new_text) == 0:
        new_text = 'error'
    return new_text

# clean
error_sampled_generations = 0
cleaned_generations = []

for generation in tqdm.tqdm(generations):
    cleaned_generation = generation
    sampled_generated_texts = cleaned_generation['sampled_generated_texts']
    cleaned_gens = []
    for sampled_generation in sampled_generated_texts:
        cleaned_gen = filter(sampled_generation)
        cleaned_gens.append(cleaned_gen)

        if cleaned_gen == 'error':
            error_sampled_generations += 1
        print(cleaned_gen)
    cleaned_generation['sampled_generated_texts'] = cleaned_gens
    cleaned_generations.append(cleaned_generation)

print('error sampled generations: ', error_sampled_generations)

# save
with open(f'{run_name}/cleaned_generations.pkl', 'wb') as record_file:
    pickle.dump(cleaned_generations, record_file)
print('Record saved to ', f'{run_name}/cleaned_generations.pkl')
