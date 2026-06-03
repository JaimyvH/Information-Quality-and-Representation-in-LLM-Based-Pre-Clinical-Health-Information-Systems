# Information Quality and Representation in LLM-Based Pre-Clinical Health Information Systems

This repository contains the working materials for a Master's thesis on how large language models perform in pre-clinical health information contexts. The project evaluates model responses to health-related prompts with attention to information quality, representation, and accessibility trade-offs between smaller and larger open-weight models.

## Repository Contents

- `AI_Health_Query_Prompt_Bank.xlsx` - initial prompt bank for the study.
- `AI_Health_Query_Prompt_Bank_filled.xlsx` - completed prompt bank used for model runs.
- `runmodelsscript.py` - resumable script for sending prompts to locally running Ollama models and saving model outputs.
- `llama13b`, `llama70b`, `qwen14b`, `qwen72b` - Ollama model definition files with shared generation settings.

## Models

The current experimental setup compares two model families at different parameter scales:

- Llama 2 13B and 70B
- Qwen 2.5 14B and 72B

This allows the thesis to compare response quality across model size and examine whether higher-quality outputs also come with higher hardware requirements.

## Running the Model Script

The script expects Ollama to be running locally at:

```text
http://localhost:11434/api/generate
```

It reads prompts from `AI_Health_Query_Prompt_Bank_filled.xlsx`, sends each prompt to the configured models, and writes the outputs to `results.xlsx` plus a CSV backup.

The script runs all prompts for one model before switching to the next model. This keeps large models loaded in Ollama longer and avoids repeated reloads.
By default, each response is allowed up to 500 generated tokens so outputs are less likely to be cut off during health-information explanations.

```powershell
python runmodelsscript.py
```

Useful options:

```powershell
python runmodelsscript.py --models llama70b --num-ctx 4096
python runmodelsscript.py --models llama13 qwen14b --repeats 4
python runmodelsscript.py --num-predict 600
python runmodelsscript.py --force
```
