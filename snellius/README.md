# Snellius Environment Setup

This folder contains Snellius-specific setup and job scripts for the thesis inference experiments.

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

If the repository is private, configure GitHub access first or copy the folder with `rsync`.

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

## 4. Run The Small Local-Equivalent Qwen3 Profile

This runs Qwen3 4B and Qwen3 8B using 4-bit loading:

```bash
sbatch snellius/run_qwen3_local.job
```

This is the recommended first real Snellius run.

## 5. Run Larger Models On H100

Use:

```bash
sbatch snellius/run_one_large_model_h100.job
```

Edit the `MODEL_ID`, `OUTPUT_FILE`, and time limit in the job script before submitting.

## Storage And Caches

The scripts use:

```bash
/scratch-shared/$USER/master-thesis/hf_cache
/scratch-shared/$USER/master-thesis/outputs
```

Hugging Face model downloads are large and should not go into `$HOME`. Scratch is temporary and not backed up. Copy final results back to `$HOME/master-thesis/results_snellius/` and then to your local machine.

## Useful Commands

Check jobs:

```bash
squeue -u $USER
```

Cancel a job:

```bash
scancel JOBID
```

Show partitions:

```bash
sinfo
```

Interactive short GPU debug session:

```bash
srun --partition=gpu_mig --gpus=1 --ntasks=1 --cpus-per-task=9 --time=00:15:00 --pty bash -i
```

## Partition Choice

- Use `gpu_mig` for Qwen3 4B/8B and quick tests.
- Use `gpu_h100` for Qwen3 14B/32B, Gemma 27B, Llama 70B, or anything that does not fit in MIG.

The SURF partition documentation lists `gpu_mig` as A100 MIG with 1 GPU slice and 60 GiB host memory, `gpu_a100` as full A100, and `gpu_h100` as full H100 with higher accounting weight.

