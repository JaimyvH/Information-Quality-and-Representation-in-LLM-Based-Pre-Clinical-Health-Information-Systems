# Model Selection Notes

The original development setup used Ollama with Llama 2 and Qwen2.5 models. For the final thesis experiment, the model runs are handled through Hugging Face Transformers on Snellius, with explicit model IDs, generation settings, seeds, output files, and four repeated generations per prompt.

## Current Design

Use two modern open-weight model families that provide multiple instruction-tuned parameter sizes:

- Qwen3: 4B and 8B for the core comparison; 14B, 32B, and 30B-A3B for a larger Snellius extension.
- Gemma 3 IT: 4B and 12B for the core comparison; 27B for a larger Snellius extension.

This gives a clearer scaling comparison than mixing older Llama 2 models with newer Qwen or Gemma models. The main statistical interpretation should compare smaller and larger models within the same family. Cross-family differences can still be reported, but should be treated descriptively because the parameter jumps and model architectures are not identical.

## Why Qwen3

Qwen3 is the preferred Qwen family because it is newer than Qwen2.5 and provides a dense size ladder including 0.6B, 1.7B, 4B, 8B, 14B, and 32B models, plus mixture-of-experts variants. This makes it well suited for testing whether quality improves with parameter count. For non-reasoning health-information answers, thinking mode should be disabled where supported so the outputs remain comparable to ordinary instruction-following models.

## Why Gemma 3

Gemma 3 offers 1B, 4B, 12B, and 27B sizes. The 4B and 12B models provide a second family-level scaling comparison, while the 27B model can serve as an upper size point if compute budget and access allow.

## Llama

Llama remains useful, but Llama 2 should be avoided for the final comparison if possible because it is older than the Qwen3 and Gemma 3 families. Llama 3.1 provides a clean 8B versus 70B comparison, but the 70B model is not part of the current run plan and can be added later only as an optional Snellius extension.

## Practical Recommendation

Core Snellius experiment:

```text
Qwen3 4B Instruct 2507
Qwen3 8B
Gemma 3 4B IT
Gemma 3 12B IT
```

Snellius extension:

```text
Qwen3 14B
Qwen3 32B
Qwen3 30B-A3B Instruct 2507
Gemma 3 27B IT
```

The thesis can frame the core study as a comparison of feasible open-weight models on Snellius and the extended runs as an expanded scaling analysis.
