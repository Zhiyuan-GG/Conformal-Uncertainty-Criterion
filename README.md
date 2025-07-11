# ConU: Conformal Uncertainty in Large Language Models with Correctness Coverage Guarantees [EMNLP 2024 Findings]
[arXiv](https://arxiv.org/abs/2407.00499)

**Authors:** Zhiyuan Wang, Jinhao Duan, Lu Cheng, Yue Zhang, Qingni Wang, Xiaoshuang Shi, Kaidi Xu, Hengtao Shen, Xiaofeng Zhu

In this paper, we introduce a novel black-box uncertainty measure, termed ConU, by sampling multiple generations and estimating the most frequent (reliable) response based on the self-consistency theory. Specifically, we combine the normalized frequency score of the most reliable response/semantic with the semantic diversity between it and response samples with other semantics. 

Based on ConU, we develop the conformal uncertainty criterion, by linking the nonconformity score with the uncertainty condition aligned with correctness (any admissible semantic), bounding the correctness coverage rate at various user-specified error rates for black-box LLMs in open-ended tasks for the first time. 

The proposed Conformal Uncertainty Criterion is implemented in this codebase. Please set the (local) paths for the model and raw data in advance.

## Data Preparing
TriviaQA (Download from hugging face https://huggingface.co/datasets/trivia_qa/tree/main/rc.nocontext)

```shell
python parse_triviaqa.py --few-shot-num 3 --max-num 2000;
```

## Generate

```shell
python clean_open_domain_qa.py --dataset triviaqa --max-length-of-generation 32 --sample True --num-beams 5 --num-generations-per-prompt 20 --top-p 0.9 --temperature 1.0;
```

## Clean irrelavant strings
```shell
python clean_open_domain_qa.py --dataset triviaqa --max-length-of-generation 32 --sample True --num-beams 5 --num-generations-per-prompt 20 --top-p 0.9 --temperature 1.0;
```

## Similarity Estimation
Compute the similarity among the sampled answers as well as their similarity to the ground-truth answer. 
```shell
python open_domain_qa_similarity.py --dataset triviaqa --max-length-of-generation 32 --sample True --num-beams 5 --num-generations-per-prompt 20 --top-p 0.9 --temperature 1.0;
```

## Conformal Procedure
```shell
python conformal_uncertainty_criterion.py --dataset triviaqa --max-length-of-generation 32 --sample True --num-beams 5 --num-generations-per-prompt 20 --top-p 0.9 --temperature 1.0 --split-ratio 0.5 --correctness-threshold 0.6 --alpha 0.1;
```
