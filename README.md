# Information Quality and Representation in LLM-Based Pre-Clinical Health Information Systems

This repository contains the working materials for a Master's thesis on how large language models perform in pre-clinical health information contexts. The project evaluates model responses to health-related prompts with attention to information quality, representation, and accessibility trade-offs between smaller and larger open-weight models.

The final inference experiments are run on Snellius with Hugging Face Transformers. Earlier Ollama-based scripts are kept only as legacy development material.

## Repository Contents

- `AI_Health_Query_Prompt_Bank_filled.xlsx` - completed prompt bank used for model runs.
- `run_transformers_prompts.py` - Hugging Face Transformers runner for the controlled Snellius experiment.
- `model_configs.json` - model profiles for the Snellius model groups.
- `MODEL_SELECTION.md` - notes on model-family selection and scaling comparisons.
- `snellius/` - Snellius setup and batch-job scripts.
- `runmodelsscript.py` - legacy Ollama runner from the earlier development phase.

## Experimental Model Groups

The current experiment compares newer open-weight model families across available parameter sizes. The main analysis focuses on within-family size comparisons rather than claiming that different families scale identically.

Core Snellius runs:

```text
Qwen/Qwen3-4B-Instruct-2507
Qwen/Qwen3-8B
google/gemma-3-4b-it
google/gemma-3-12b-it
```

Extended Snellius runs, if compute budget and model access allow:

```text
Qwen/Qwen3-14B
Qwen/Qwen3-32B
Qwen/Qwen3-30B-A3B-Instruct-2507
google/gemma-3-27b-it
```

Llama models are intentionally left out of the current run plan and can be added later as an optional extension.

## Repeated Runs

Each model is run four times per prompt by default. The output spreadsheet stores the first run under the model label and later repeats as `_run2`, `_run3`, and `_run4`, with matching `_seconds` timing columns.

## Running On Snellius

See `snellius/README.md` for the full workflow. The main jobs are:

```bash
sbatch snellius/run_qwen3_core.job
sbatch snellius/run_gemma3_core.job
sbatch snellius/run_qwen3_extended_h100.job
sbatch snellius/run_gemma3_27b_h100.job
```

Qwen3 thinking mode is disabled by default where the tokenizer supports it, keeping outputs closer to ordinary instruction-following health-information answers. Use `--enable-thinking` only for a separate reasoning-focused robustness check.
