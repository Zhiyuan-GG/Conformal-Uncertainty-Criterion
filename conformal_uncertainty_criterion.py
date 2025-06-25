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
parser.add_argument('--split-ratio', type=float, default=0.5, help='for splitting calibration and test set')
parser.add_argument('--correctness-threshold', type=float, default=0.7, help='for correctness evaluation')
parser.add_argument('--alpha', type=float, default=0.2, help='risk level')
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
# load similarity scores
with open(f'{run_name}/similarity_scores.pkl', 'rb') as record_file:
    similarity_scores = pickle.load(record_file)

similarity_dict, similarity_for_correctness = similarity_scores
# ----------------------------------------------------------------------------------------------------------------------
# assumption: at least one acceptable generation
applied_generations = []
for generation in generations:
    id = generation['id']
    similarity_for_correctness_list = similarity_for_correctness[id]
    if max(similarity_for_correctness_list) >= args.correctness_threshold:
        applied_generations.append(generation)

total_num = len(applied_generations)
num_cal = int(total_num * args.split_ratio)
calibration_set = random.sample(applied_generations, num_cal)
test_set = [generation for generation in applied_generations if generation not in calibration_set]
print('Applied num: ', total_num)
print('Calibration num: ', len(calibration_set))
print('Test num: ', len(test_set))
# ----------------------------------------------------------------------------------------------------------------------
# nonconformity scores in calibration set
nonconformity_scores = []
for cal_data in tqdm.tqdm(calibration_set):
    cal_id = cal_data['id']
    # for NS
    sampled_similarity_dict = similarity_dict[cal_id]
    # for correctness check
    sampled_correctness_similarity_list = similarity_for_correctness[cal_id]
    # frequency
    sampled_generation_frequency_scores = []
    for sampled_idx in sampled_similarity_dict:
        sampled_generation_frequency_scores.append(
            np.sum(np.array(sampled_similarity_dict[sampled_idx]) >= args.correctness_threshold) / len(
                sampled_similarity_dict[sampled_idx]))
    # numpy
    sampled_generation_frequency_scores = np.array(sampled_generation_frequency_scores)
    # ------------------------------------------------------------------------------------------------------------------
    normalized_prob = sampled_generation_frequency_scores / np.sum(sampled_generation_frequency_scores)
    predictive_entropy = -np.sum(normalized_prob[normalized_prob > 0] * np.log(normalized_prob[normalized_prob > 0]))
    # ------------------------------------------------------------------------------------------------------------------
    most_reliable_generation_idx = sampled_correctness_similarity_list.index(max(sampled_correctness_similarity_list))

    most_reliable_generation_similarity_scores = np.array(sampled_similarity_dict[most_reliable_generation_idx])

    most_reliable_generation_frequency = normalized_prob[most_reliable_generation_idx]
    most_reliable_generation_semantic_diversity = np.dot(normalized_prob,
                                                         most_reliable_generation_similarity_scores) / len(
        normalized_prob)

    nonconformity_score = 1 -  most_reliable_generation_frequency
    nonconformity_scores.append(nonconformity_score)
# ----------------------------------------------------------------------------------------------------------------------
N = len(nonconformity_scores)
q_level = np.ceil((N + 1) * (1 - args.alpha)) / N
q_hat = np.quantile(nonconformity_scores, q_level, method='higher')

miscoverage_num = 0
total_set_size = 0
total_confidence_weighted_ss = 0
for test_data in tqdm.tqdm(test_set):
    test_id = test_data['id']
    sampled_similarity_dict = similarity_dict[test_id]
    sampled_correctness_similarity_list = np.array(similarity_for_correctness[test_id])

    sampled_generation_frequency_scores = []
    for sampled_idx in sampled_similarity_dict:
        sampled_generation_frequency_scores.append(
            np.sum(np.array(sampled_similarity_dict[sampled_idx]) >= args.correctness_threshold) / len(
                sampled_similarity_dict[sampled_idx]))
    sampled_generation_frequency_scores = np.array(sampled_generation_frequency_scores)
    normalized_prob = sampled_generation_frequency_scores / np.sum(sampled_generation_frequency_scores)
    # prediction set
    prediction_set = []
    for sampled_idx in sampled_similarity_dict:
        sampled_frequency_score = normalized_prob[sampled_idx]
        sampled_semantic_diversity = np.dot(np.array(sampled_similarity_dict[sampled_idx]),
                                            normalized_prob) / len(
            normalized_prob)
        sampled_nonconformity_score = 1 -  sampled_frequency_score
        if sampled_nonconformity_score <= q_hat:
            prediction_set.append(sampled_idx)
    # set size
    set_size = len(prediction_set)
    total_set_size += set_size

    if not np.any(sampled_correctness_similarity_list[prediction_set] >= args.correctness_threshold):
        miscoverage_num += 1
miscoverage_rate = miscoverage_num / len(test_set)
print('EMR: ', miscoverage_rate)
print('APSS: ', total_set_size / len(test_set))


