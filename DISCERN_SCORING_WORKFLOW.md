# DISCERN Scoring Workflow

This project uses DISCERN scoring to evaluate generated pre-clinical health information. The primary dataset for scoring is created by converting model-output spreadsheets from wide format to one response per row.

## Prepare A Scoring Workbook

Use:

```bash
python prepare_discern_scoring.py results_snellius/results_qwen3_core_snellius.xlsx --output results_snellius/discern_scoring_qwen3_core.xlsx
```

After Gemma has been regenerated successfully, include both families:

```bash
python prepare_discern_scoring.py \
  results_snellius/results_qwen3_core_snellius.xlsx \
  results_snellius/results_gemma3_core_snellius.xlsx \
  --output results_snellius/discern_scoring_core_models.xlsx
```

The workbook contains:

- `scoring_items`: one randomized row per valid model response.
- `errors`: technical failures, such as tracebacks or empty responses.
- `rubric`: DISCERN question descriptions.
- `readme`: scoring notes.

The first model run is stored under the model label. Later repeats use `_run2`, `_run3`, and `_run4` in the original result files.

## Error Handling

Technical errors should not be treated as health-information answers. They should either be regenerated or reported separately as technical failures. Weak, incomplete, or medically poor answers should remain in the scoring set because they represent model quality.

## Gemma Regeneration Note

If Gemma outputs contain CUDA tracebacks rather than health-information responses, regenerate the Gemma result file after pulling the latest code. The Transformers runner keeps the same substantive decoding settings but enables logits renormalization and invalid-value removal during generation, and the Gemma Snellius jobs use bfloat16 4-bit compute on A100/H100 hardware. This avoids treating technical GPU sampling failures as model answers.

## AutoDiscern-Adapted Automated Scoring

The AutoDiscern repository from Krauthammer Lab demonstrates that automated DISCERN-style assessment is feasible, but it is not directly plug-and-play for this thesis because the repository does not include the underlying data or trained experiment objects and was designed for online health webpages rather than generated LLM responses.

For this thesis, we use an AutoDiscern-adapted scoring setup:

1. The input unit is changed from website HTML to an LLM prompt-response pair.
2. The document `content` field is represented by the generated response text.
3. Prompt metadata is retained as document metadata.
4. The scoring target is expanded to the full 16-item DISCERN framework.
5. The unavailable AutoDiscern trained classifier is replaced by a deterministic rubric-guided judge model.

Run the adapted scorer on Snellius:

```bash
sbatch snellius/run_autodiscern_adapted_qwen3.job
```

By default this uses `meta-llama/Llama-3.1-70B-Instruct` as the judge model because it is separate from the evaluated Qwen3 and Gemma 3 model families. If access to the Llama model is unavailable, set `JUDGE_MODEL_ID` to another instruction-tuned model before submitting:

```bash
JUDGE_MODEL_ID=Qwen/Qwen3-32B sbatch snellius/run_autodiscern_adapted_qwen3.job
```

The output is saved to:

```text
/scratch-shared/$USER/master-thesis/outputs/autodiscern_adapted_scores_qwen3_core.xlsx
```

and copied to:

```text
$HOME/master-thesis/results_snellius/
```
