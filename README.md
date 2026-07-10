# Extending SigExt with Comparative Encoder Architectures

This repository branch extends the official implementation of the EMNLP 2024 paper:

**Salient Information Prompting to Steer Content in Prompt-based Abstractive Summarization**

Original authors: Lei Xu, Mohammed Asad Karim, Saket Dingliwal, and Aparna Elangovan.

The original SigExt repository introduces a keyphrase-based prompting framework for abstractive summarization. Our work builds on this framework by establishing a comparative framework for encoder architectures to evaluate how varying architectural paradigms impact long-document keyphrase extraction.

## Project Overview

Identifying salient information in extensive source documents depends heavily on the underlying encoder's ability to process long sequences efficiently. This extension contrasts legacy sparse-attention structures against highly optimized, dense-attention modern architectures.

This project studies the interaction between:

- sparse and global attention mechanisms;
- dense, high-capacity modern encoding;
- computational viability (VRAM footprint and latency);
- factual alignment in downstream abstractive summaries.

The experiments focus primarily on two summarization domains:

- **CNN/DailyMail**, containing general-domain news articles;
- **arXiv**, containing long scientific papers and abstracts.

The evaluated encoder architectures are:

- Longformer-large;
- BigBird-large;
- ModernBERT-large.

## Main Contributions

Compared with the original SigExt implementation, this branch introduces the following changes.

### Architectural Evaluation Framework

The extraction pipeline was extended to benchmark diverse design paradigms for long-document processing:

- **Sparse/Global Attention**: Longformer and BigBird are evaluated to assess how structural token management impacts sequence understanding.
- **Optimized Dense Attention**: ModernBERT is integrated to evaluate the advantages of high-capacity, native dense processing without sparse complications.

### Factual Alignment and Metric Analysis

The evaluation setup expands beyond standard lexical overlap metrics (ROUGE-1, ROUGE-2, ROUGE-L) to analyze deep text quality:

- **Factual Faithfulness**: Integration of AlignScore to measure true structural and factual alignment.
- **Qualitative Comparison**: Evaluates how the choice of keyphrase extractor (sparse vs. dense) directly influences the factual reliability of the final generated summaries.

### Hyperparameter Ablation Studies

The repository includes scripts to test the stability and hidden operational costs of sparse models:

- **Sparsity Ablation**: Systematically varies BigBird's random attention blocks (R) and block sizes (B).
- **Sensitivity Analysis**: Measures how structural hyperparameters affect summarization quality and generation lengths, contrasting this with the hyperparameter-free nature of modern dense architectures.

### Operational and Computational Profiling

Hardware tracking is integrated directly into the inference pipeline to assess real-world deployment viability:

- **VRAM Tracking**: Monitors peak memory footprint during high-throughput cycles to assess the operational efficiency of each architecture.
- **Latency & Throughput**: Tracks execution speed per document to determine the scalability of deploying these extractors in production-level pipelines.

## Running the Code

The dataset preparation remains unchanged from the original SigExt implementation. Use the updated extraction tools to run comparative encoder experiments and extract hardware metrics.

### Run Keyphrase Extraction with ModernBERT

```bash
python3 src/extract_keyphrases.py \
  --extractor_model modernbert_large \
  --dataset arxiv \
  --dataset_dir experiments/arxiv_dataset/ \
  --output_dir experiments/arxiv/modernbert/ \
  --batch_size 8
