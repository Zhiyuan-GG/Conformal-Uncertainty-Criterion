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
from sentence_transformers import CrossEncoder

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
                    default=r'D:\projects\cache_model\Qwen2.5-3B-Instruct',
                    help='local path of generative llm')
parser.add_argument('--dataset', default='triviaqa')
parser.add_argument('--max-length-of-generation', type=int, default=128)
parser.add_argument('--sample', type=bool, default=True, help='sampled or the most likely')  ##
parser.add_argument('--num-beams', type=int, default=5, help='for the most likely generation')
parser.add_argument('--num-generations-per-prompt', type=int, default=10, help='for sampling')
parser.add_argument('--top-p', type=float, default=0.9, help='for sampling')
parser.add_argument('--temperature', type=float, default=1.0, help='for sampling')
# ----------------------------------------------------------------------------------------------------------------------
# for semantic similarity
parser.add_argument('--similarity-model',
                    default=r'D:\projects\cache_model\stsb-roberta-large',
                    help='sentence similarity')
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
with open(f'{run_name}/cleaned_generations.pkl', 'rb') as record_file:
    generations = pickle.load(record_file)

# for sentence similarity
similarity_model = CrossEncoder(model_name=args.similarity_model, num_labels=1)
# ----------------------------------------------------------------------------------------------------------------------
similarity_dict = dict()
similarity_for_correctness = dict()
for generation in tqdm.tqdm(generations):
    id = generation['id']
    gt_answer = generation['answer']
    sampled_generated_texts = generation['sampled_generated_texts']
    question = generation['question']
    # ------------------------------------------------------------------------------------------------------------------
    similarity_dict[id] = dict()  # {0: [], 1: [], ...}
    for i, gen in enumerate(sampled_generated_texts):
        similarity_dict[id][i] = []
        for j, gen_temp in enumerate(sampled_generated_texts):
            qa_1 = question + ' ' + gen
            qa_2 = question + ' ' + gen_temp
            if j == i:
                similarity_dict[id][i].append(1.0)
            elif j > i:
                similarity_dict[id][i].append(similarity_model.predict([qa_1, qa_2]))
            elif j < i:
                similarity_dict[id][i].append(similarity_dict[id][j][i])
        print(f'id-{id} --- similarity-{i}: ', similarity_dict[id][i])
    # ------------------------------------------------------------------------------------------------------------------
    similarity_to_gt_list = []
    for sampled_gen in sampled_generated_texts:
        qa_1 = question + ' ' + gt_answer
        qa_2 = question + ' ' + sampled_gen
        similarity_to_gt_list.append(similarity_model.predict([qa_1, qa_2]))
    similarity_for_correctness[id] = similarity_to_gt_list
    print(f'id-{id} --- similarity for correctness: ', similarity_for_correctness[id])

# similarity_dict: sampled responses之间
# similarity_for_correctness: sampled responses与gt之间
results = [similarity_dict, similarity_for_correctness]
# save
with open(f'{run_name}/similarity_scores.pkl', 'wb') as record_file:
    pickle.dump(results, record_file)
print('Record saved to ', f'{run_name}/similarity_scores.pkl')

