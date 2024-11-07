# MILU: A Multi-task Indic Language Understanding Benchmark


[![ArXiv](https://img.shields.io/badge/arXiv-2411.02538-b31b1b.svg)](https://arxiv.org/abs/2411.02538)     [![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-yellow)](https://huggingface.co/datasets/ai4bharat/MILU) [![CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

## Overview

MILU (Multi-task Indic Language Understanding Benchmark) is a comprehensive evaluation dataset designed to assess the performance of Large Language Models (LLMs) across 11 Indic languages. It spans 8 domains and 42 subjects, reflecting both general and culturally specific knowledge from India.

This repository contains code for evaluating language models on the MILU benchmark using the [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness) framework.



## Usage

##### Prerequisites

- Python 3.7+
- `lm-eval-harness` library
- HuggingFace Transformers
- vLLM (optional, for faster inference)

1. Clone this repository:

```bash
git clone --depth 1 https://github.com/AI4Bharat/MILU.git
cd MILU
pip install -e .
```

2. Request access to the HuggingFace 🤗 dataset [here](https://huggingface.co/datasets/ai4bharat/MILU).

3. Set up your environment variables:

```bash
export HF_HOME=/path/to/HF_CACHE/if/needed
export HF_TOKEN=YOUR_HUGGINGFACE_TOKEN
```

The following languages are supported for MILU:
- Bengali
- English
- Gujarati
- Hindi
- Kannada
- Malayalam
- Marathi
- Odia
- Punjabi
- Tamil
- Telugu

### HuggingFace Evaluation

For HuggingFace models, you may use the following sample command:

```bash
lm_eval --model hf \
    --model_args 'pretrained=google/gemma-2-27b-it,temperature=0.0,top_p=1.0,parallelize=True' \
    --tasks milu \
    --batch_size auto:40 \  
    --log_samples \
    --output_path $EVAL_OUTPUT_PATH \
    --max_batch_size 64 \
    --num_fewshot 5 \
    --apply_chat_template
```

### vLLM Evaluation

For vLLM-compatible models, you may use the following sample command:

```bash
lm_eval --model vllm \
    --model_args 'pretrained=meta-llama/Llama-3.2-3B,tensor_parallel_size=$N_GPUS' \
    --gen_kwargs 'temperature=0.0,top_p=1.0' \
    --tasks milu \
    --batch_size auto \
    --log_samples \
    --output_path $EVAL_OUTPUT_PATH
```

### Single Language Evaluation

To evaluate your Model on a specific language, modify the `--tasks` parameter:

```bash
--tasks milu_English
```

Replace `English` with the available language (e.g., Odia, Hindi, etc.).

### Evaluation Tips & Observations

1. Make sure to use `--apply_chat_template` for Instruction-fine-tuned models, to format the prompt correctly.
2. vLLM generally works better with Llama models, while Gemma models work better with HuggingFace.
3. If vLLM encounters out-of-memory errors, try reducing `max_gpu_utilization` else switch to HuggingFace.
4. For HuggingFace, use `--batch_size=auto:<n_batch_resize_tries>` to re-select the batch size multiple times.
5. When using vLLM, pass generation kwargs in the `--gen_kwargs` flag. For HuggingFace, include them in `model_args`.


## Key Features

- **11 Indian Languages**: Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu, and English
- **Domains**: 8 diverse domains including Arts & Humanities, Social Sciences, STEM, and more
- **Subjects**: 42 subjects covering a wide range of topics
- **Questions**: ~85,000 multiple-choice questions
- **Cultural Relevance**: Incorporates India-specific knowledge from regional and state-level examinations

## Dataset Statistics

| Language | Total Questions | Translated Questions | Avg Words Per Question |
|----------|-----------------|----------------------|------------------------|
| Bengali  | 7138            | 1601                 | 15.72                  |
| Gujarati | 5327            | 2755                 | 16.69                  |
| Hindi    | 15450           | 115                  | 20.63                  |
| Kannada  | 6734            | 1522                 | 12.83                  |
| Malayalam| 4670            | 1534                 | 12.82                  |
| Marathi  | 7424            | 1235                 | 18.8                   |
| Odia     | 5025            | 1452                 | 15.63                  |
| Punjabi  | 4363            | 2341                 | 19.9                   |
| Tamil    | 7059            | 1524                 | 13.32                  |
| Telugu   | 7847            | 1298                 | 16.13                  |
| English  | 14036           | -                    | 22.01                  |
| **Total**| **85073**       | **15377**            | **16.77** (avg)        |



## Dataset Structure

### Test Set
The test set consists of the MILU (Multi-task Indic Language Understanding) benchmark, which contains approximately 85,000 multiple-choice questions across 11 Indic languages.

### Validation Set
The dataset includes a separate validation set of 9,157 samples that can be used for few-shot examples during evaluation. This validation set was created by sampling questions from each of the 42 subjects.

### Subjects spanning MILU


| Domain | Subjects |
|--------|----------|
| Arts & Humanities | Architecture and Design, Arts and Culture, Education, History, Language Studies, Literature and Linguistics, Media and Communication, Music and Performing Arts, Religion and Spirituality |
| Business Studies | Business and Management, Economics, Finance and Investment |
| Engineering & Tech | Energy and Power, Engineering, Information Technology, Materials Science, Technology and Innovation, Transportation and Logistics |
| Environmental Sciences | Agriculture, Earth Sciences, Environmental Science, Geography |
| Health & Medicine | Food Science, Health and Medicine |
| Law & Governance | Defense and Security, Ethics and Human Rights, Law and Ethics, Politics and Governance |
| Math and Sciences | Astronomy and Astrophysics, Biology, Chemistry, Computer Science, Logical Reasoning, Mathematics, Physics |
| Social Sciences | Anthropology, International Relations, Psychology, Public Administration, Social Welfare and Development, Sociology, Sports and Recreation |




## Evaluation

We evaluated 45 different LLMs on MILU, including:

- Closed proprietary models (e.g., GPT-4o, Gemini-1.5)
- Open-source multilingual models
- Language-specific fine-tuned models

Key findings:

- GPT-4o achieved the highest average accuracy at 72%
- Open multilingual models outperformed language-specific fine-tuned models
- Models performed better in high-resource languages compared to low-resource ones
- Performance was lower in culturally relevant areas (e.g., Arts & Humanities) compared to general fields like STEM

For detailed results and analysis, please refer to our [paper](https://arxiv.org/abs/2411.02538).

## Citation

If you use MILU in your research, please cite our paper:

```bibtex
@article{verma2024milu,
  title   = {MILU: A Multi-task Indic Language Understanding Benchmark},
  author  = {Sshubam Verma and Mohammed Safi Ur Rahman Khan and Vishwajeet Kumar and Rudra Murthy and Jaydeep Sen},
  year    = {2024},
  journal = {arXiv preprint arXiv: 2411.02538}
}
```

## License

This dataset is released under the [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Contact

For any questions or feedback, please contact:
- Sshubam Verma (sshubamverma@ai4bharat.org)
- Mohammed Safi Ur Rahman Khan (safikhan@ai4bharat.org)
- Rudra Murthy (rmurthyv@in.ibm.com)
- Vishwajeet Kumar (vishk024@in.ibm.com)

## Links

- [GitHub Repository 💻](https://github.com/AI4Bharat/MILU)
- [Paper 📄](https://arxiv.org/abs/2411.02538)

