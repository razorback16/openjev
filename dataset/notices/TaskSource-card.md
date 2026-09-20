---
language:
- en
license: apache-2.0
size_categories:
- 1M<n<10M
task_categories:
- text-generation
- text-classification
- token-classification
- zero-shot-classification
pretty_name: tasksource-instruct
dataset_info:
  features:
  - name: inputs
    dtype: string
  - name: targets
    dtype: string
  - name: task
    dtype: string
  splits:
  - name: train
    num_bytes: 3351995517.351683
    num_examples: 5314383
  - name: test
    num_bytes: 89780918.3443312
    num_examples: 150287
  - name: validation
    num_bytes: 87728387.29075804
    num_examples: 142950
  download_size: 1886645135
  dataset_size: 3529504822.9867725
tags:
- instructions
- instruction-tuning
- instruction-finetuning
- flan
- promptsource
- tasksource
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-*
  - split: test
    path: data/test-*
  - split: validation
    path: data/validation-*
---
# Dataset Card for "tasksource-instruct-v0" (TSI)


Multi-task instruction-tuning data recasted from 485 of the [tasksource](https://github.com/sileod/tasksource) datasets.
Dataset size is capped at 30k examples per task to foster task diversity.

```python
!pip install tasksource, pandit
import tasksource, pandit
df = tasksource.list_tasks(instruct=True).sieve(id=lambda x: 'mmlu' not in x)
for tasks in df.id:
  yield tasksource.load_task(task,instruct=True,max_rows=30_000,max_rows_eval=200)
```

https://github.com/sileod/tasksource

## How it differs from flan-v2

TSI is HuggingFace-centric and based on tasksource, a curated collection of HF datasets. It can be scaled to much more examples.
tasksource is focused on discriminative tasks (Classification/TokenClassification/MultipleChoice). The coverage on discriminative tasks is greater than flan.
List of tasks [here](https://github.com/sileod/tasksource/blob/main/tasks.md). Examples of tasks not in Flan V2 include Dynasent (adversarial sentiment analysis), Dynahate (adversarial hate speech detection, discriminative babi, epistemic logic, ruletaker, veridicality, discourse relation prediction, dozens of interesting natural language inference datasets...

TSI answers are mostly short answers to multiple-choice questions, but they target a wide array of problems.
TSI is reasoning intensive, while some flan tasks are not necessarily specific (e.g. generating hypothesis based on premise for NLI).
We explicitly mention that answers should not have explanations, to prevent biasing models toward short answers when using other instruction datasets.

`flan-v2` and `tasksource-instruct` can be combined to improve the reasoning capabilities of LLM. 

## Contact and citation:
damien.sileo@inria.fr

https://arxiv.org/abs/2301.05948
```
@inproceedings{sileo-2024-tasksource,
    title = "tasksource: A Large Collection of {NLP} tasks with a Structured Dataset Preprocessing Framework",
    author = "Sileo, Damien",
    editor = "Calzolari, Nicoletta  and
      Kan, Min-Yen  and
      Hoste, Veronique  and
      Lenci, Alessandro  and
      Sakti, Sakriani  and
      Xue, Nianwen",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.1361/",
    pages = "15655--15684",
}
```