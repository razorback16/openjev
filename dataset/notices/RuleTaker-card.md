---
dataset_info:
  features:
  - name: context
    dtype: string
  - name: question
    dtype: string
  - name: label
    dtype: string
  - name: config
    dtype: string
  splits:
  - name: train
    num_bytes: 252209259
    num_examples: 480152
  - name: dev
    num_bytes: 39591713
    num_examples: 75872
  - name: test
    num_bytes: 80649163
    num_examples: 151911
  download_size: 34172740
  dataset_size: 372450135
license: apache-2.0
language:
- en
---
# Dataset Card for "ruletaker"
https://github.com/allenai/ruletaker

```
@inproceedings{ruletaker2020,
  title     = {Transformers as Soft Reasoners over Language},
  author    = {Clark, Peter and Tafjord, Oyvind and Richardson, Kyle},
  booktitle = {Proceedings of the Twenty-Ninth International Joint Conference on
               Artificial Intelligence, {IJCAI-20}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Christian Bessiere},
  pages     = {3882--3890},
  year      = {2020},
  month     = {7},
  note      = {Main track},
  doi       = {10.24963/ijcai.2020/537},
  url       = {https://doi.org/10.24963/ijcai.2020/537},
}

```