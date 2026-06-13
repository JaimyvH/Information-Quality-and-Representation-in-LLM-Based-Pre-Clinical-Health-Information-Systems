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

## Automated Scoring

The Auto-DISCERN repository from Krauthammer Lab demonstrates that automated DISCERN-style assessment is feasible, but it is not directly plug-and-play for this thesis because the repository does not include the underlying data or trained experiment objects and was designed for online health webpages rather than generated LLM responses.

For this thesis, automated scoring should therefore be treated as an auxiliary or robustness method unless it is validated against expert scoring on a subset of responses. A practical workflow is:

1. Use `prepare_discern_scoring.py` to create blinded scoring items.
2. Obtain expert DISCERN scores for a subset or full set of responses.
3. Run an automated scorer on the same rows.
4. Compare automated scores with expert scores before using them for hypothesis testing.

