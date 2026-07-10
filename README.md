# Domain-Adaptive Prompting and Extraction Architectures for Abstractive Summarization with Large Language Models

This work studies the impact of prompting strategies and information extraction architectures on abstractive summarization with Large Language Models. Starting from the SigExt framework, we evaluate the base model, keyword-guided prompting, Chain-of-Thought, eventually combined with SigExt, and entity-relation reasoning on CNN/DailyMail and arXiv using Mistral, Qwen, and LLaMA. We also compare Longformer, BigBird, and ModernBERT as extraction modules. The results show that SigExt performs best on CNN/DailyMail, while reasoning-based prompts are more effective for selected models on arXiv. ModernBERT achieves the best overall trade-off between factual alignment and computational efficiency. These results highlight the importance of adapting both prompting and extraction strategies to the target domain and language model.

## Repository Structure (How to navigate the project)

To keep the codebase clean and organized, we developed a common baseline and then built two parallel extensions. You can switch between the different versions of the project using the **Branch** dropdown menu in the top-left corner.

Here is how the repository is structured:

### 1. Current Branch: `main` 
This branch is a fork/clone of the official [Amazon Science SigExt](https://github.com/amazon-science/SigExt) implementation. 
We used this codebase as our foundational baseline and extended its functionalities as part of our project.

### 2. Branch: `[different-models]` [[Click here to view this branch](https://github.com/ZeiX-P/DNLP_project/tree/different-models)]
This branch contains the first project variant, focusing on:
* **information extraction module** used for input conditioning, replacing the original extractor with alternative encoder-based models.

### 3. Branch: `[llm-models+cot]` [[Click here to view this branch](https://github.com/ZeiX-P/DNLP_project/tree/llm-models%2Bcot)]
This branch contains the second project variant, focusing on:
* **Prompting strategy**, comparing baseline prompting with progressively more structured reasoning-based approaches, including Chain-of-Thought and entity-relation guided prompting.
