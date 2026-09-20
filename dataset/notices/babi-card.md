---
annotations_creators:
- expert-generated
language_creators:
- crowdsourced
language:
- en
license: bsd
multilinguality:
- monolingual
size_categories:
- 1K<n<10K
source_datasets:
- original
task_categories:
- text-classification
task_ids:
- natural-language-inference
pretty_name: babi_nli
tags:
- logical reasoning
- nli
- natural-language-inference
- reasoning
- logic
dataset_info:
- config_name: agents-motivations
  features:
  - name: premise
    dtype: string
  - name: hypothesis
    dtype: string
  - name: label
    dtype:
      class_label:
        names:
          '0': not-entailed
          '1': entailed
  - name: idx
    dtype: int32
  splits:
  - name: train
    num_bytes: 194721
    num_examples: 1000
  - name: validation
    num_bytes: 97602
    num_examples: 500
  - name: test
    num_bytes: 101683
    num_examples: 500
  download_size: 79703
  dataset_size: 394006
- config_name: basic-coreference
  features:
  - name: premise
    dtype: string
  - name: hypothesis
    dtype: string
  - name: label
    dtype:
      class_label:
        names:
          '0': not-entailed
          '1': entailed
  - name: idx
    dtype: int32
  splits:
  - name: train
    num_bytes: 247359
    num_examples: 1000
  - name: validation
    num_bytes: 125770
    num_examples: 500
  - name: test
    num_bytes: 122865
    num_examples: 500
  download_size: 98403
  dataset_size: 495994
- config_name: basic-deduction
  features:
  - name: premise
    dtype: string
  - name: hypothesis
    dtype: string
  - name: label
    dtype:
      class_label:
        names:
          '0': not-entailed
          '1': entailed
  - name: idx
    dtype: int32
  splits:
  - name: train
    num_bytes: 225519
    num_examples: 1000
  - name: validation
    num_bytes: 112981
    num_examples: 500
  - name: test
    num_bytes: 112862
    num_examples: 500
  download_size: 49023
  dataset_size: 451362
- config_name: basic-induction
  features:
  - name: premise
    dtype: string
  - name: hypothesis
    dtype: string
  - name: label
    dtype:
      class_label:
        names:
          '0': not-entailed
          '1': entailed
  - name: idx
    dtype: int32
  splits:
  - name: train
    num_bytes: 188319
    num_examples: 1000
  - name: validation
    num_bytes: 94081
    num_examples: 500
  - name: test
    num_bytes: 94090
    num_examples: 500
  download_size: 85153
  dataset_size: 376490
- config_name: compound-coreference
  features:
  - name: premise
    dtype: string
  - name: hypothesis
    dtype: string
  - name: label
    dtype:
      class_label:
        names:
          '0': not-entailed
          '1': entailed
  - name: idx
    dtype: int32
  splits:
  - name: train
    num_bytes: 282687
    num_examples: 1000
  - name: validation
    num_bytes: 139161
    num_examples: 500
  - name: test
    num_bytes: 143924
    num_examples: 500
  download_size: 109694
  dataset_size: 565772
- config_name: conjunction
  features:
  - name: premise
    dtype: string
  - name: hypothesis
    dtype: string
  - name: label
    dtype:
      class_label:
        names:
          '0': not-entailed
          '1': entailed
  - name: idx
    dtype: int32
  splits:
  - name: train
    num_bytes: 284487
    num_examples: 1000
  - name: validation
    num_bytes: 141534
    num_examples: 500
  - name: test
    num_bytes: 142683
    num_examples: 500
  download_size: 118704
  dataset_size: 568704
- config_name: counting
  features:
  - name: premise
    dtype: string
  - name: hypothesis
    dtype: string
  - name: label
    dtype:
      class_label:
        names:
          '0': not-entailed
          '1': entailed
  - name: idx
    dtype: int32
  splits:
  - name: train
    num_bytes: 325802
    num_examples: 1000
  - name: validation
    num_bytes: 154693
    num_examples: 500
  - name: test
    num_bytes: 154519
    num_examples: 500
  download_size: 119022
  dataset_size: 635014
- config_name: indefinite-knowledge
  features:
  - name: premise
    dtype: string
  - name: hypothesis
    dtype: string
  - name: label
    dtype:
      class_label:
        names:
          '0': not-entailed
          '1': entailed
  - name: idx
    dtype: int32
  splits:
  - name: train
    num_bytes: 227391
    num_examples: 1000
  - name: validation
    num_bytes: 113140
    num_examples: 500
  - name: test
    num_bytes: 113298
    num_examples: 500
  download_size: 96252
  dataset_size: 453829
- config_name: lists-sets
  features:
  - name: premise
    dtype: string
  - name: hypothesis
    dtype: string
  - name: label
    dtype:
      class_label:
        names:
          '0': not-entailed
          '1': entailed
  - name: idx
    dtype: int32
  splits:
  - name: train
    num_bytes: 301224
    num_examples: 1000
  - name: validation
    num_bytes: 149567
    num_examples: 500
  - name: test
    num_bytes: 153024
    num_examples: 500
  download_size: 119319
  dataset_size: 603815
- config_name: path-finding
  features:
  - name: premise
    dtype: string
  - name: hypothesis
    dtype: string
  - name: label
    dtype:
      class_label:
        names:
          '0': not-entailed
          '1': entailed
  - name: idx
    dtype: int32
  splits:
  - name: train
    num_bytes: 253540
    num_examples: 1000
  - name: validation
    num_bytes: 126723
    num_examples: 500
  - name: test
    num_bytes: 126695
    num_examples: 500
  download_size: 105706
  dataset_size: 506958
- config_name: positional-reasoning
  features:
  - name: premise
    dtype: string
  - name: hypothesis
    dtype: string
  - name: label
    dtype:
      class_label:
        names:
          '0': not-entailed
          '1': entailed
  - name: idx
    dtype: int32
  splits:
  - name: train
    num_bytes: 158135
    num_examples: 1000
  - name: validation
    num_bytes: 79034
    num_examples: 500
  - name: test
    num_bytes: 78998
    num_examples: 500
  download_size: 34376
  dataset_size: 316167
- config_name: simple-negation
  features:
  - name: premise
    dtype: string
  - name: hypothesis
    dtype: string
  - name: label
    dtype:
      class_label:
        names:
          '0': not-entailed
          '1': entailed
  - name: idx
    dtype: int32
  splits:
  - name: train
    num_bytes: 218297
    num_examples: 1000
  - name: validation
    num_bytes: 110967
    num_examples: 500
  - name: test
    num_bytes: 107212
    num_examples: 500
  download_size: 88932
  dataset_size: 436476
- config_name: single-supporting-fact
  features:
  - name: premise
    dtype: string
  - name: hypothesis
    dtype: string
  - name: label
    dtype:
      class_label:
        names:
          '0': not-entailed
          '1': entailed
  - name: idx
    dtype: int32
  splits:
  - name: train
    num_bytes: 223761
    num_examples: 1000
  - name: validation
    num_bytes: 112784
    num_examples: 500
  - name: test
    num_bytes: 111569
    num_examples: 500
  download_size: 91968
  dataset_size: 448114
- config_name: size-reasoning
  features:
  - name: premise
    dtype: string
  - name: hypothesis
    dtype: string
  - name: label
    dtype:
      class_label:
        names:
          '0': not-entailed
          '1': entailed
  - name: idx
    dtype: int32
  splits:
  - name: train
    num_bytes: 314623
    num_examples: 1000
  - name: validation
    num_bytes: 155731
    num_examples: 500
  - name: test
    num_bytes: 154007
    num_examples: 500
  download_size: 53727
  dataset_size: 624361
- config_name: three-arg-relations
  features:
  - name: premise
    dtype: string
  - name: hypothesis
    dtype: string
  - name: label
    dtype:
      class_label:
        names:
          '0': not-entailed
          '1': entailed
  - name: idx
    dtype: int32
  splits:
  - name: train
    num_bytes: 598485
    num_examples: 1000
  - name: validation
    num_bytes: 312900
    num_examples: 500
  - name: test
    num_bytes: 324225
    num_examples: 500
  download_size: 243096
  dataset_size: 1235610
- config_name: three-supporting-facts
  features:
  - name: premise
    dtype: string
  - name: hypothesis
    dtype: string
  - name: label
    dtype:
      class_label:
        names:
          '0': not-entailed
          '1': entailed
  - name: idx
    dtype: int32
  splits:
  - name: train
    num_bytes: 1479097
    num_examples: 1000
  - name: validation
    num_bytes: 783450
    num_examples: 500
  - name: test
    num_bytes: 735719
    num_examples: 500
  download_size: 558073
  dataset_size: 2998266
- config_name: time-reasoning
  features:
  - name: premise
    dtype: string
  - name: hypothesis
    dtype: string
  - name: label
    dtype:
      class_label:
        names:
          '0': not-entailed
          '1': entailed
  - name: idx
    dtype: int32
  splits:
  - name: train
    num_bytes: 379567
    num_examples: 1000
  - name: validation
    num_bytes: 188913
    num_examples: 500
  - name: test
    num_bytes: 187157
    num_examples: 500
  download_size: 152925
  dataset_size: 755637
- config_name: two-arg-relations
  features:
  - name: premise
    dtype: string
  - name: hypothesis
    dtype: string
  - name: label
    dtype:
      class_label:
        names:
          '0': not-entailed
          '1': entailed
  - name: idx
    dtype: int32
  splits:
  - name: train
    num_bytes: 122565
    num_examples: 1000
  - name: validation
    num_bytes: 61228
    num_examples: 500
  - name: test
    num_bytes: 61170
    num_examples: 500
  download_size: 54533
  dataset_size: 244963
- config_name: two-supporting-facts
  features:
  - name: premise
    dtype: string
  - name: hypothesis
    dtype: string
  - name: label
    dtype:
      class_label:
        names:
          '0': not-entailed
          '1': entailed
  - name: idx
    dtype: int32
  splits:
  - name: train
    num_bytes: 507085
    num_examples: 1000
  - name: validation
    num_bytes: 261742
    num_examples: 500
  - name: test
    num_bytes: 244322
    num_examples: 500
  download_size: 194986
  dataset_size: 1013149
- config_name: yes-no-questions
  features:
  - name: premise
    dtype: string
  - name: hypothesis
    dtype: string
  - name: label
    dtype:
      class_label:
        names:
          '0': not-entailed
          '1': entailed
  - name: idx
    dtype: int32
  splits:
  - name: train
    num_bytes: 230579
    num_examples: 1000
  - name: validation
    num_bytes: 115915
    num_examples: 500
  - name: test
    num_bytes: 113022
    num_examples: 500
  download_size: 94399
  dataset_size: 459516
configs:
- config_name: agents-motivations
  data_files:
  - split: train
    path: agents-motivations/train-*
  - split: validation
    path: agents-motivations/validation-*
  - split: test
    path: agents-motivations/test-*
- config_name: basic-coreference
  data_files:
  - split: train
    path: basic-coreference/train-*
  - split: validation
    path: basic-coreference/validation-*
  - split: test
    path: basic-coreference/test-*
- config_name: basic-deduction
  data_files:
  - split: train
    path: basic-deduction/train-*
  - split: validation
    path: basic-deduction/validation-*
  - split: test
    path: basic-deduction/test-*
- config_name: basic-induction
  data_files:
  - split: train
    path: basic-induction/train-*
  - split: validation
    path: basic-induction/validation-*
  - split: test
    path: basic-induction/test-*
- config_name: compound-coreference
  data_files:
  - split: train
    path: compound-coreference/train-*
  - split: validation
    path: compound-coreference/validation-*
  - split: test
    path: compound-coreference/test-*
- config_name: conjunction
  data_files:
  - split: train
    path: conjunction/train-*
  - split: validation
    path: conjunction/validation-*
  - split: test
    path: conjunction/test-*
- config_name: counting
  data_files:
  - split: train
    path: counting/train-*
  - split: validation
    path: counting/validation-*
  - split: test
    path: counting/test-*
- config_name: indefinite-knowledge
  data_files:
  - split: train
    path: indefinite-knowledge/train-*
  - split: validation
    path: indefinite-knowledge/validation-*
  - split: test
    path: indefinite-knowledge/test-*
- config_name: lists-sets
  data_files:
  - split: train
    path: lists-sets/train-*
  - split: validation
    path: lists-sets/validation-*
  - split: test
    path: lists-sets/test-*
- config_name: path-finding
  data_files:
  - split: train
    path: path-finding/train-*
  - split: validation
    path: path-finding/validation-*
  - split: test
    path: path-finding/test-*
- config_name: positional-reasoning
  data_files:
  - split: train
    path: positional-reasoning/train-*
  - split: validation
    path: positional-reasoning/validation-*
  - split: test
    path: positional-reasoning/test-*
- config_name: simple-negation
  data_files:
  - split: train
    path: simple-negation/train-*
  - split: validation
    path: simple-negation/validation-*
  - split: test
    path: simple-negation/test-*
- config_name: single-supporting-fact
  data_files:
  - split: train
    path: single-supporting-fact/train-*
  - split: validation
    path: single-supporting-fact/validation-*
  - split: test
    path: single-supporting-fact/test-*
- config_name: size-reasoning
  data_files:
  - split: train
    path: size-reasoning/train-*
  - split: validation
    path: size-reasoning/validation-*
  - split: test
    path: size-reasoning/test-*
- config_name: three-arg-relations
  data_files:
  - split: train
    path: three-arg-relations/train-*
  - split: validation
    path: three-arg-relations/validation-*
  - split: test
    path: three-arg-relations/test-*
- config_name: three-supporting-facts
  data_files:
  - split: train
    path: three-supporting-facts/train-*
  - split: validation
    path: three-supporting-facts/validation-*
  - split: test
    path: three-supporting-facts/test-*
- config_name: time-reasoning
  data_files:
  - split: train
    path: time-reasoning/train-*
  - split: validation
    path: time-reasoning/validation-*
  - split: test
    path: time-reasoning/test-*
- config_name: two-arg-relations
  data_files:
  - split: train
    path: two-arg-relations/train-*
  - split: validation
    path: two-arg-relations/validation-*
  - split: test
    path: two-arg-relations/test-*
- config_name: two-supporting-facts
  data_files:
  - split: train
    path: two-supporting-facts/train-*
  - split: validation
    path: two-supporting-facts/validation-*
  - split: test
    path: two-supporting-facts/test-*
- config_name: yes-no-questions
  data_files:
  - split: train
    path: yes-no-questions/train-*
  - split: validation
    path: yes-no-questions/validation-*
  - split: test
    path: yes-no-questions/test-*
---

# bAbi_nli

bAbI tasks recasted as natural language inference.
https://github.com/facebookarchive/bAbI-tasks

tasksource recasting code:
https://colab.research.google.com/drive/1J_RqDSw9iPxJSBvCJu-VRbjXnrEjKVvr?usp=sharing

```bibtex
@article{weston2015towards,
  title={Towards ai-complete question answering: A set of prerequisite toy tasks},
  author={Weston, Jason and Bordes, Antoine and Chopra, Sumit and Rush, Alexander M and Van Merri{\"e}nboer, Bart and Joulin, Armand and Mikolov, Tomas},
  journal={arXiv preprint arXiv:1502.05698},
  year={2015}
}
```