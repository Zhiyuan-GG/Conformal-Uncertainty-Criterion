import argparse
import json
import random

import numpy as np
import tqdm
import re
import os

import accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM

import torch
import pandas as pd
import datasets
from datasets import Dataset

# ----------------------------------------------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--generate-model',
                    default=r'D:\projects\cache_model\openchat_3.5',
                    help='local path of generative llm downloaded from Hugging Face')
parser.add_argument('--row-data-path',
                    default=r'../row_data/triviaqa/validation-00000-of-00001.parquet',
                    help='local path of row dataset')
parser.add_argument('--data-dir',
                    default='../datasets',
                    help='save parsed dataset')
parser.add_argument('--cache-dir',
                    default='../cache',
                    help='cache model from hugging face')
parser.add_argument('--few-shot-num',
                    default=3,
                    help='for few-shot prompt')
parser.add_argument('--max-num',
                    default=4000,
                    help='for save')
args = parser.parse_args()
# ----------------------------------------------------------------------------------------------------------------------
# model_name for path of saved parsed dataset
model_name = args.generate_model.split('\\')[-1] if '\\' in args.generate_model else args.generate_model
print('Generative LLM: ', model_name)
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
# set cuda device 0,1
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
os.environ["HF_DATASETS_CACHE"] = args.cache_dir
# ----------------------------------------------------------------------------------------------------------------------
# for input_ids length (allowed by llm)
generative_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.generate_model,
                                                     local_files_only=True,
                                                     # resume_download=True,
                                                     # cache_dir=arg.cache_dir,
                                                     # use_auth_token="your_token",
                                                     # proxies='xxx',
                                                     # trust_remote_code=True,
                                                     use_fast=False)
generative_llm = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=args.generate_model,
                                                      local_files_only=True,
                                                      torch_dtype=torch.float16,
                                                      # resume_download=True,
                                                      # cache_dir=arg.cache_dir,
                                                      # use_auth_token="your_token",
                                                      # proxies='xxx',
                                                      # trust_remote_code=True,
                                                      device_map="auto")  # require accelerate
max_input_ids_length = generative_llm.config.max_position_embeddings
print('LLM max input ids length: ', max_input_ids_length)
# ----------------------------------------------------------------------------------------------------------------------
# load row data (download from hugging face https://huggingface.co/datasets/trivia_qa/tree/main/rc.nocontext)
df = pd.read_parquet(args.row_data_path)
dict_list = df.to_dict(orient='records')
print('Num samples (or question-answer pairs): ', len(dict_list))  # 17944
# 'question'   : str,
# 'question_id': str,
# 'answer'     : dict{'value'}
# ----------------------------------------------------------------------------------------------------------------------
# prompt engineer
instruction = "### System:\nThis is a bot that correctly answers questions.\n\n"

idx_for_few_shot_prompt = random.sample(range(0, len(dict_list)), args.few_shot_num)
print(idx_for_few_shot_prompt)  # [1067, 14053, 15812]

few_shot_prompt = ''
for idx in idx_for_few_shot_prompt:
    question = dict_list[idx]['question']
    answer = dict_list[idx]['answer']['value']
    # examine ASCII for encode and decode
    temp = ''
    assert question.isascii() and answer.isascii()

    few_shot_prompt += '### User:\n' + question + '\n### Assistant:\n' + answer + '\n\n'
# ----------------------------------------------------------------------------------------------------------------------
dataset = {}

dataset['prompt'] = []
dataset['question'] = []
dataset['answer'] = []
dataset['id'] = []
# ----------------------------------------------------------------------------------------------------------------------
# parse
applied_qa_pairs = 0
for sample_idx, sample in enumerate(tqdm.tqdm(dict_list)):
    if sample_idx not in idx_for_few_shot_prompt:
        question = sample['question']
        answer = sample['answer']['value']
        if question.isascii() and answer.isascii():
            prompt = instruction + few_shot_prompt + '### User:\n' + question + '\n### Assistant:\n'
            input_ids = generative_tokenizer.encode(prompt)
            if len(input_ids) < max_input_ids_length:
                applied_qa_pairs += 1
                print(prompt)
                exit()
                dataset['prompt'].append(prompt)
                dataset['question'].append(question)
                dataset['answer'].append(answer)
                dataset['id'].append(dict_list[sample_idx]['question_id'] + '_' + str(sample_idx))

                if applied_qa_pairs == args.max_num:
                    break

print('Applied question-answer pairs: ', applied_qa_pairs)
# ----------------------------------------------------------------------------------------------------------------------
# save
dataset_df = pd.DataFrame.from_dict(dataset)
dataset = Dataset.from_pandas(dataset_df)
dataset.save_to_disk(f'{args.data_dir}/triviaqa_{model_name}')
# ----------------------------------------------------------------------------------------------------------------------
# check answer length
len_list = []
for answer in dataset['answer']:
    answer_ids = generative_tokenizer.encode(answer)
    len_list.append(len(answer_ids))

print('Max answer length: ', max(len_list))