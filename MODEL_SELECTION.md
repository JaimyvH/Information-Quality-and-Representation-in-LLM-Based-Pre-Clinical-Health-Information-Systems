# Model Selection Notes

The original Ollama setup used Llama 2 13B/70B and Qwen2.5 14B/72B. For the final thesis experiment, it is cleaner to use Hugging Face Transformers with explicitly recorded model IDs, generation settings, seeds, and output files.

## Recommended Local Design

Use two modern open-weight model families that provide multiple instruction-tuned parameter sizes:

- Qwen3: 4B and 8B locally; 14B, 32B, and 30B-A3B if Snellius access becomes available.
- Gemma 3 IT: 4B and 12B locally; 27B if Snellius or sufficient local quantization is available.

This gives a clearer scaling comparison than mixing Llama 2 13B with Qwen models, because the selected families are more recent and provide multiple size points within the same generation.

## Why Qwen3

Qwen3 is the preferred Qwen family because it is newer than Qwen2.5 and provides a dense size ladder including 0.6B, 1.7B, 4B, 8B, 14B, and 32B models, plus mixture-of-experts variants. This makes it well suited for testing whether quality improves with parameter count. For non-reasoning health-information answers, thinking mode should be disabled where supported so the outputs remain comparable to ordinary instruction-following models.

## Why Gemma 3

Gemma 3 offers 1B, 4B, 12B, and 27B sizes. The 4B and 12B models are more realistic for local experimentation, while the 27B model can serve as an upper size point if additional compute becomes available.

## Llama

Llama remains useful, but Llama 2 should be avoided for the final comparison if possible because it is older than the Qwen3 and Gemma 3 families. Llama 3.1 provides a clean 8B versus 70B comparison, but the 70B model is not realistically feasible on the local machine without heavy quantization or offloading. It is better suited as an optional Snellius extension.

## Practical Recommendation

Primary local experiment:

```text
Qwen3 4B Instruct 2507
Qwen3 8B
Gemma 3 4B IT
Gemma 3 12B IT
```

Snellius extension, if compute is granted:

```text
Qwen3 14B
Qwen3 32B
Qwen3 30B-A3B Instruct 2507
Llama 3.1 70B Instruct
Gemma 3 27B IT
```

The thesis can frame the local study as a comparison of feasible open-weight models and the Snellius study, if available, as an expanded scaling analysis.
