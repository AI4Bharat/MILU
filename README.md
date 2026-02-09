# MILU: A Multi-task Indic Language Understanding Benchmark


[![ArXiv](https://img.shields.io/badge/arXiv-2411.02538-b31b1b.svg)](https://arxiv.org/abs/2411.02538)     [![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Datasets-yellow)](https://huggingface.co/datasets/ai4bharat/MILU) [![CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

## Overview

MILU (Multi-task Indic Language Understanding Benchmark) is a comprehensive evaluation dataset designed to assess the performance of Large Language Models (LLMs) across 11 Indic languages. It spans 8 domains and 41 subjects, reflecting both general and culturally specific knowledge from India.

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
- **Subjects**: 41 subjects covering a wide range of topics
- **Questions**: ~80,000 multiple-choice questions
- **Cultural Relevance**: Incorporates India-specific knowledge from regional and state-level examinations

## Dataset Statistics

| Language   | Total Questions | Translated Questions | Avg Words Per Question |
|------------|----------------|----------------------|------------------------|
| Bengali    | 6638           | 1601                 | 15.12                  |
| Gujarati   | 4827           | 2755                 | 16.12                  |
| Hindi      | 14837          | 115                  | 20.61                  |
| Kannada    | 6234           | 1522                 | 12.42                  |
| Malayalam  | 4321           | 3354                 | 12.39                  |
| Marathi    | 6924           | 1235                 | 18.76                  |
| Odia       | 4525           | 3100                 | 14.96                  |
| Punjabi    | 4099           | 3411                 | 19.26                  |
| Tamil      | 6372           | 1524                 | 13.14                  |
| Telugu     | 7304           | 1298                 | 15.71                  |
| English    | 13536          | -                    | 22.07                  |
| **Total**  | **79617**      | **19915**            | **16.41** (avg)        |



## Dataset Structure

### Test Set
The test set consists of the MILU (Multi-task Indic Language Understanding) benchmark, which contains approximately 85,000 multiple-choice questions across 11 Indic languages.

### Validation Set
The dataset includes a separate validation set of 8,933 samples that can be used for few-shot examples during evaluation. This validation set was created by sampling questions from each of the 41 subjects.

### Subjects spanning MILU


| Domain | Subjects |
|--------|----------|
| Arts & Humanities | Architecture and Design, Arts and Culture, Education, History, Language Studies, Literature and Linguistics, Media and Communication, Music and Performing Arts, Religion and Spirituality |
| Business Studies | Business and Management, Economics, Finance and Investment |
| Engineering & Tech | Energy and Power, Engineering, Information Technology, Materials Science, Technology and Innovation, Transportation and Logistics |
| Environmental Sciences | Agriculture, Earth Sciences, Environmental Science, Geography |
| Health & Medicine | Food Science, Health and Medicine |
| Law & Governance | Defense and Security, Ethics and Human Rights, Law and Ethics, Politics and Governance |
| Math and Sciences | Astronomy and Astrophysics, Biology, Chemistry, Computer Science, Logical Reasoning, Physics |
| Social Sciences | Anthropology, International Relations, Psychology, Public Administration, Social Welfare and Development, Sociology, Sports and Recreation |




## Citation

If you use MILU in your work, please cite us:

```bibtex
@inproceedings{verma-etal-2025-milu,
    title = "{MILU}: A Multi-task {I}ndic Language Understanding Benchmark",
    author = "Verma, Sshubam  and
      Khan, Mohammed Safi Ur Rahman  and
      Kumar, Vishwajeet  and
      Murthy, Rudra  and
      Sen, Jaydeep",
    editor = "Chiruzzo, Luis  and
      Ritter, Alan  and
      Wang, Lu",
    booktitle = "Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)",
    month = apr,
    year = "2025",
    address = "Albuquerque, New Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.naacl-long.507/",
    doi = "10.18653/v1/2025.naacl-long.507",
    pages = "10076--10132",
    ISBN = "979-8-89176-189-6",
    abstract = "Evaluating Large Language Models (LLMs) in low-resource and linguistically diverse languages remains a significant challenge in NLP, particularly for languages using non-Latin scripts like those spoken in India. Existing benchmarks predominantly focus on English, leaving substantial gaps in assessing LLM capabilities in these languages. We introduce MILU, a Multi-task Indic Language Understanding Benchmark, a comprehensive evaluation benchmark designed to address this gap. MILU spans 8 domains and 41 subjects across 11 Indic languages, reflecting general and culturally specific knowledge. With an India-centric design, incorporates material from regional and state-level examinations, covering topics such as local history, arts, festivals, and laws, alongside standard subjects like science and mathematics. We evaluate over 42 LLMs, and find that current LLMs struggle with MILU, with GPT-4o achieving the highest average accuracy at 74 percent. Open multilingual models outperform language-specific fine-tuned models, which perform only slightly better than random baselines. Models also perform better in high resource languages as compared to low resource ones. Domain-wise analysis indicates that models perform poorly in culturally relevant areas like Arts and Humanities, Law and Governance compared to general fields like STEM. To the best of our knowledge, MILU is the first of its kind benchmark focused on Indic languages, serving as a crucial step towards comprehensive cultural evaluation. All code, benchmarks, and artifacts are publicly available to foster open research."
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
- [Hugging Face Dataset 🤗](https://huggingface.co/datasets/ai4bharat/MILU)

