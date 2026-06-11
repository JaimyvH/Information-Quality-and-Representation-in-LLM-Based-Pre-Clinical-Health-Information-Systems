# Snellius Environment Setup

This folder contains Snellius-specific setup and job scripts for the thesis inference experiments. The final model runs are intended to be executed on Snellius, not on a local workstation.

## 1. Copy Or Clone The Repository

On Snellius, put the repository in:

```bash
$HOME/master-thesis
```

For example:

```bash
git clone https://github.com/JaimyvH/Information-Quality-and-Representation-in-LLM-Based-Pre-Clinical-Health-Information-Systems.git $HOME/master-thesis
cd $HOME/master-thesis
```

If the repository already exists, update it with:

```bash
cd $HOME/master-thesis
git pull
```

## 2. Create The Python Environment

Submit the setup job:

```bash
sbatch snellius/setup_env.job
```

This creates:

```bash
$HOME/.venvs/master-thesis-transformers
```

The environment installs PyTorch, Transformers, Accelerate, BitsAndBytes, and the spreadsheet dependencies.

## 3. Run A GPU Smoke Test

After the setup job finishes:

```bash
sbatch snellius/gpu_smoke_test.job
```

Check the output in `logs/`. The test should print that CUDA is available and show the assigned GPU.

## 4. Core Model Runs

Each job uses four repeats per prompt by default.

Run Qwen3 4B and Qwen3 8B:

```bash
sbatch snellius/run_qwen3_core.job
```

Run Gemma 3 4B IT and Gemma 3 12B IT:

```bash
sbatch snellius/run_gemma3_core.job
```

The Gemma models may require accepting the model license on Hugging Face and setting `HF_TOKEN` in the Snellius environment before running.

## 5. Extended H100 Runs

Run larger Qwen3 models:

```bash
sbatch snellius/run_qwen3_extended_h100.job
```

Run Gemma 3 27B IT:

```bash
sbatch snellius/run_gemma3_27b_h100.job
```

The helper script `run_one_large_model_h100.job` can be used for a single custom model, but the dedicated jobs above should be preferred for the planned thesis runs.

## Resuming And Repeats

The runner resumes from an existing output file. If a previous result file already contains run 1, a four-repeat job will skip existing valid cells and fill the missing `_run2`, `_run3`, and `_run4` columns.

The output spreadsheet stores:

```text
model_label
model_label_seconds
model_label_run2
model_label_run2_seconds
model_label_run3
model_label_run3_seconds
model_label_run4
model_label_run4_seconds
```

## Storage And Caches

The scripts use:

```bash
/scratch-shared/$USER/master-thesis/hf_cache
/scratch-shared/$USER/master-thesis/outputs
```

Hugging Face model downloads are large and should not go into `$HOME`. Scratch is temporary and not backed up. The scripts copy final results back to `$HOME/master-thesis/results_snellius/`; download final files from either that folder or from scratch.

## Useful Commands

Check jobs:

```bash
squeue -u $USER
```

Cancel a job:

```bash
scancel JOBID
```

Show newest logs:

```bash
ls -lt logs/*.out
tail -n 80 logs/NEWEST_LOG_FILE.out
```

Interactive short GPU debug session:

```bash
srun --partition=gpu_mig --gpus=1 --ntasks=1 --cpus-per-task=9 --time=00:15:00 --pty bash -i
```

## Partition Choice

- Use `gpu_mig` for Qwen3 4B/8B, Gemma 3 4B/12B, and quick tests.
- Use `gpu_h100` for Qwen3 14B/32B/30B-A3B, Gemma 3 27B, or anything that does not fit in MIG.

The SURF partition documentation lists `gpu_mig` as A100 MIG with 1 GPU slice and 60 GiB host memory, `gpu_a100` as full A100, and `gpu_h100` as full H100 with higher accounting weight.
