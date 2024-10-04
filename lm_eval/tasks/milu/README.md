# Task-name

### Paper

Title: `paper titles goes here`

Abstract: `link to paper PDF or arXiv abstract goes here`

`Short description of paper / benchmark goes here:`

Homepage: `homepage to the benchmark's website goes here, if applicable`


### Citation

```
BibTeX-formatted citation goes here
```

### Groups, Tags, and Tasks

#### Groups

* `group_name`: `Short description`

#### Tags

* `tag_name`: `Short description`

#### Tasks

* `task_name`: `1-sentence description of what this particular task does`
* `task_name2`: ...

### Checklist

For adding novel benchmarks/datasets to the library:
* [ ] Is the task an existing benchmark in the literature?
  * [ ] Have you referenced the original paper that introduced the task?
  * [ ] If yes, does the original paper provide a reference implementation? If so, have you checked against the reference implementation and documented how to run such a test?


If other tasks on this dataset are already supported:
* [ ] Is the "Main" variant of this task clearly denoted?
* [ ] Have you provided a short sentence in a README on what each new variant adds / evaluates?
* [ ] Have you noted which, if any, published evaluation setups are matched by this variant?


# How to run
---

### MILU Evaluation Script

This repository contains a script for evaluating language models on the Indic Massive Multitask Language Understanding (MILU) benchmark using the `lm-eval-harness` framework.

#### Features

- Supports evaluation of HuggingFace and vLLM models
- Configurable number of few-shot examples (0, 1, 5)
- Automatic batch size selection
- Logging of samples and outputs
- Multi-GPU support

##### Prerequisites

- Python 3.7+
- `lm-eval-harness` library
- HuggingFace Transformers
- vLLM (optional, for faster inference)

## Usage

1. Clone this repository:

```bash
git clone --depth 1 https://github.com/AI4Bharat/MILU.git
cd MILU
pip install -e .
```

2. Set up your environment variables:

```bash
export HF_HOME=/path/to/HF_CACHE/if/needed
export HF_TOKEN=YOUR_HUGGINGFACE_TOKEN
```


## Configuration

You can customize the evaluation by modifying the following variables in the script:

- `MODEL_NAME`: The name of the model from HuggingFace Model Hub
- `EVAL_OUTPUT_PATH`: The directory to store evaluation results
- `shots`: An array of few-shot settings to evaluate (default: 0, 1, 5)

## Supported Languages
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

## HuggingFace Evaluation

For HuggingFace models, the script uses the following command:

```bash
lm_eval --model hf \
    --model_args 'pretrained=$MODEL_NAME,temperature=0.0,top_p=1.0,parallelize=True' \
    --tasks milu \
    --batch_size auto:40 \  
    --log_samples \
    --output_path $EVAL_OUTPUT_PATH \
    --max_batch_size 64
```

## vLLM Evaluation

For vLLM-compatible models, use the following command:

```bash
lm_eval --model vllm \
    --model_args 'pretrained=meta-llama/Llama-3.2-3B,tensor_parallel_size=$N_GPUS' \
    --gen_kwargs 'temperature=0.0,top_p=1.0' \
    --tasks milu \
    --batch_size auto \
    --log_samples \
    --output_path $EVAL_OUTPUT_PATH
```

## Single Language Evaluation

To evaluate your on a specific language, modify the `--tasks` parameter:

```bash
--tasks milu_English
```

Replace `English` with the available language (e.g., Odia, Hindi, etc.).

## Tips & Observations

1. vLLM generally works better with Llama models, while Gemma models work better with HuggingFace.
2. If vLLM encounters out-of-memory errors, switch to HuggingFace.
3. For HuggingFace, use `--batch_size=auto:<n_batch_resize_tries>` to re-select the batch size multiple times.
4. When using vLLM, pass generation kwargs in the `--gen_kwargs` flag. For HuggingFace, include them in `model_args`.
