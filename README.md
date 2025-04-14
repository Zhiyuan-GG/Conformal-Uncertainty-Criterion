# Conformal-Uncertainty-Criterion
### ConU: Conformal Uncertainty in Large Language Models with Correctness Coverage Guarantees (Findings of EMNLP 2024)

In this paper, we introduce a novel black-box uncertainty measure, termed ConU, by sampling multiple generations and estimating the most frequent (reliable) response based on the self-consistency theory. Specifically, we combine the normalized frequency score of the most reliable response/semantic with the semantic diversity between it and response samples with other semantics. 

Based on ConU, we develop the conformal uncertainty criterion, by linking the nonconformity score with the uncertainty condition aligned with correctness (any admissible semantic), bounding the correctness coverage rate at various user-specified error rates for black-box LLMs in open-ended tasks for the first time. 

