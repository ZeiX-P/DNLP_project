# Extending SigExt with Multi-LLM and Reasoning-Based Prompting

This repository extends the official implementation of the EMNLP 2024 paper:

**Salient Information Prompting to Steer Content in Prompt-based Abstractive Summarization**

Original authors: Lei Xu, Mohammed Asad Karim, Saket Dingliwal, and Aparna Elangovan.

The original SigExt repository introduces a keyphrase-based prompting framework for abstractive summarization. Our work builds on this framework by adding support for multiple local instruction-tuned Large Language Models and by introducing reasoning-based prompting strategies.

## Project Overview

Large Language Models can generate high-quality abstractive summaries without task-specific fine-tuning. However, their outputs are highly sensitive to prompt design, target domain, summary length constraints, and model architecture.

This project studies the interaction between:

- Large Language Models;
- salient-information prompting;
- Chain-of-Thought prompting;
- entity-relation reasoning;
- dataset-specific prompt design.

The experiments focus primarily on two summarization domains:

- **CNN/DailyMail**, containing general-domain news articles;
- **arXiv**, containing long scientific papers and abstracts.

The supported language models are:

- Mistral-7B-Instruct;
- Qwen2.5-7B-Instruct;
- LLaMA-3.1-8B-Instruct.

## Main Contributions

Compared with the original SigExt implementation, this branch introduces the following changes.

### Multi-LLM Support

The summarization pipeline was extended to support local inference with:

- Mistral;
- Qwen;
- LLaMA.

Each model uses its corresponding native chat-template format.

### Reasoning-Based Prompting

The project implements and compares several prompting strategies:

- **Zero-shot**: standard summarization without extracted keywords;
- **SigExt**: summarization guided by salient keyphrases;
- **Chain-of-Thought**: document analysis followed by summary generation;
- **Chain-of-Thought + SigExt**: reasoning combined with extracted keyphrases;
- **Entity-Relation CoT**: entity and relation extraction followed by structured summary generation.

### Dataset-Specific Prompting

The prompts are adapted to the characteristics of each dataset.

For CNN/DailyMail, the prompts focus on:

- main actors;
- location and time;
- central events;
- immediate outcomes;
- concise journalistic summaries.

For arXiv, the prompts focus on:

- research problem;
- methodology;
- main findings;
- contribution;
- limitations;
- abstract-style generation.

## Running the Code

The preprocessing pipeline is unchanged from the original SigExt implementation. Follow the original SigExt instructions to:

1. prepare the dataset;
2. train the keyphrase extractor;
3. extract and save the salient keyphrases.

Then use `summarization_batch.py` to run the extended prompting and multi-LLM experiments.

### Run Zero-Shot Summarization

```bash
python3 src/summarization_batch.py \
  --model_name llama \
  --kw_strategy disable \
  --dataset cnn \
  --dataset_dir experiments/cnn_dataset_with_keyphrase/ \
  --output_dir experiments/cnn/llama/zero_shot/ \
  --batch_size 10
