---
language:
- en
license:
- odc-by
- cc-by-4.0
- apache-2.0
- other
size_categories:
- 100K<n<1M
task_categories:
- text-generation
configs:
- config_name: default
  data_files:
  - split: chat
    path: data/chat.jsonl
  - split: instruction_following
    path: data/instruction_following.jsonl
tags:
- chat
- instruction-following
- supervised-fine-tuning
- sft
- synthetic
- text
- Nemotron_3_Ultra
---

## Dataset Description:

The Nemotron-Instruction-Following-Chat-v3 dataset is designed to strengthen multi-turn, interactive capabilities, including open-ended chat and precise instruction following.

The chat subset uses human written prompts from sources like [lmarena](https://huggingface.co/lmarena-ai), [lmsys](https://huggingface.co/datasets/lmsys/lmsys-chat-1m), and [wildchat](https://huggingface.co/datasets/allenai/WildChat-1M) as seed prompts. Responses are generated with [GLM-5](https://huggingface.co/zai-org/GLM-5). Multiple responses are sampled from the model and the best response as judged by pairwise comparisons using [Qwen3-Nemotron-235B-A22B-GenRM-2603](https://huggingface.co/nvidia/Qwen3-Nemotron-235B-A22B-GenRM-2603) is used as the target assistant response at that turn. Each conversation is further extended to multiple turns by user simulation with GLM-5. For multi-turn robustness, a randomly sampled response at a given turn is used as the context for extension instead of using the best judged response. Only the last assistant turn in each sample should hence be used for training.

**Note**: For the chat split, only the last assistant turn in each conversation should be used for training.

<mark>Certain prompts in the chat split are sourced externally from [lmsys-chat-1m](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) and [WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M). To avoid redistributing those seed prompts, rows from these sources have the initial system message content and first user message content set to `null` in `data/chat.jsonl`.</mark> Users who have access to the original source datasets can run `prepare_chat_prompts.py` to reconstruct a local version with those prompts restored. Access to `lmsys/lmsys-chat-1m` may require Hugging Face authentication and acceptance of the upstream dataset terms.

To restore the withheld prompts locally, first set a Hugging Face token that has access to `lmsys/lmsys-chat-1m`, then run:

```bash
export HF_TOKEN=<your_huggingface_token>
python prepare_chat_prompts.py \
  --input data/chat.jsonl \
  --output data/chat.with_prompts.jsonl \
  --token true
```

The instruction following data is generated with the same pipeline as [Nemotron-SFT-Instruction-Following-Chat-v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Instruction-Following-Chat-v2) using GPT-OSS-120B (medium effort). We further include prompts from [tulu-3-sft-personas-instruction-following](https://huggingface.co/datasets/allenai/tulu-3-sft-personas-instruction-following) and generate responses using GPT-OSS-120B.

This dataset is ready for commercial/non-commercial use.

## Dataset Owner(s):
NVIDIA Corporation

## Dataset Creation Date:
Created on: 06/04/2026  
Last Modified on: 06/04/2026

## Version:
Nemotron-SFT-Instruction-Following-Chat-v3

Previous Version(s):
- The chat split is intended to replace the chat data released in [Nemotron-Instruction-Following-Chat-v1](https://huggingface.co/datasets/nvidia/Nemotron-Instruction-Following-Chat-v1).
- The instruction following split is intended to be used in addition to the data in [Nemotron-SFT-Instruction-Following-Chat-v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Instruction-Following-Chat-v2).

## License/Terms of Use: 
This dataset is licensed under CC BY 4.0 and the Open Data Commons Attribution License (ODC-By).

## Intended Usage:

This dataset is intended to be used by the community to continue to improve the Instruction Following and Chat capabilities of models. The data may be freely used to train and evaluate.

### Chat Data Usage
- The chat subset is intended as a replacement of the chat data in [Nemotron-Instruction-Following-Chat-v1](https://huggingface.co/datasets/nvidia/Nemotron-Instruction-Following-Chat-v1).
- Only the last assistant turn in each sample should hence be used for training.
- Users can also use the dataset for non-reasoning/thinking training by removing the "reasoning_content" fields from assistant responses in the messages.

### Instruction Following Data Usage
- It is recommended to blend the instruction following subset with the data in [Nemotron-SFT-Instruction-Following-Chat-v2](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Instruction-Following-Chat-v2).
- Users can also use the dataset for non-reasoning/thinking training by removing the "reasoning_content" fields from assistant responses in the messages.

## Dataset Characterization

**Data Collection Method**  
* Hybrid: Human, Synthetic, Automated

**Labeling Method**
* Hybrid: Human, Synthetic, Automated

## Dataset Format
Modality: Text  
Format: JSONL  
Structure: Text + Metadata

## Dataset Quantification
| Subset                     | Samples | Storage |
|----------------------------|---------|---------|
| Chat                       | 637K    | 16G     |
| Instruction Following      | 249K    | 3G      |
| Total                      | 887K    | 19G     |

## Ethical Considerations:
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. Developers should work with their internal developer teams to ensure this dataset meets requirements for the relevant industry and use case and addresses unforeseen product misuse. Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).