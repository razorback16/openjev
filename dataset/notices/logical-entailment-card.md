---
license: apache-2.0
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
  - split: validation
    path: data/validation-*
dataset_info:
  features:
  - name: A
    dtype: string
  - name: B
    dtype: string
  - name: E
    dtype: int64
  - name: H1
    dtype: int64
  - name: H2
    dtype: int64
  - name: H3
    dtype: int64
  - name: label
    dtype: string
  splits:
  - name: train
    num_bytes: 9803153
    num_examples: 99876
  - name: test
    num_bytes: 550241
    num_examples: 5000
  - name: validation
    num_bytes: 548346
    num_examples: 5000
  download_size: 2505053
  dataset_size: 10901740
---
https://github.com/google-deepmind/logical-entailment-dataset
```
@inproceedings{
evans2018can,
title={Can Neural Networks Understand Logical Entailment?},
author={Richard Evans and David Saxton and David Amos and Pushmeet Kohli and Edward Grefenstette},
booktitle={International Conference on Learning Representations},
year={2018},
url={https://openreview.net/forum?id=SkZxCk-0Z},
}
```
                   
