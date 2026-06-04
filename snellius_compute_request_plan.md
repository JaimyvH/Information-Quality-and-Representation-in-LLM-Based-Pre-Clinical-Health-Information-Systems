# Snellius / SURF Computing Time Preparation

## Recommendation

The NWO call "Rekentijd Nationale Computerfaciliteiten 2026" is intended for large computing-time requests. For Snellius, the threshold is more than 1,000,000 SBU. The thesis workload described here is likely below that threshold unless the study is expanded substantially, for example by running many large models, repeated generations, or extensive robustness checks.

Recommended next step: contact SURF first and ask whether this should be submitted as a small computing-time request through SURF or as a large NWO request. The call explicitly states that applicants should consult a SURF advisor before submitting the application.

## Project Title

Assessing Information Quality and Representation in LLM-Based Pre-Clinical Health Information Systems

## Draft Scientific Summary

This project evaluates the quality and accessibility of large language model generated pre-clinical health information. Users increasingly consult conversational AI systems to understand symptoms, medication use, treatment options, and self-care decisions before seeking professional medical advice. The thesis investigates whether larger open-weight language models produce higher-quality health information than smaller models, and whether such quality improvements are practically accessible to stakeholders with limited computational resources.

The computational work consists of controlled inference experiments using open-weight instruction-tuned language models from modern model families such as Qwen3, Gemma 3, and, if compute permits, Llama 3.1. A curated prompt bank of health-related queries will be submitted to each model under fixed generation settings. Model outputs will be scored using an adapted DISCERN framework for consumer health information quality. The resulting dataset will be analysed using paired model comparisons and dimension-level quality analysis.

Access to GPU-based compute is needed because local hardware is insufficient for reliable and reproducible inference with larger models such as Qwen3 14B/32B, Gemma 3 27B, and Llama 3.1 70B. Snellius would make it possible to run these models under controlled, transparent, and reproducible conditions using Hugging Face Transformers, fixed seeds, documented quantization settings, and logged hardware/runtime metadata.

## Draft Scientific Case

The project contributes to research on the role of generative AI systems as information intermediaries in health-related decision-making. Existing work often evaluates LLMs using diagnostic accuracy, benchmark performance, or hallucination rates. This thesis instead focuses on structured consumer health information quality, using DISCERN to assess whether responses are reliable, transparent, balanced, and useful for pre-clinical information seeking.

The central scientific question is whether increasing model scale improves the quality of LLM-generated pre-clinical health information, and whether such improvements are sufficiently large to justify the additional computational burden. The planned computational runs compare multiple open-weight model families at different parameter scales. Smaller models can be run locally, but larger models require GPU resources beyond the available local RTX 3060 with 12 GB VRAM.

The requested compute resources will be used only for inference, not for model training or fine-tuning. Each prompt will be submitted to each selected model under fixed generation settings. The outputs, generation time, model configuration, quantization settings, and hardware metadata will be stored for later statistical analysis. This enables a reproducible comparison of model size, information quality, and computational accessibility.

## Draft Local Facilities Justification

The local machine available for the thesis has an NVIDIA GeForce RTX 3060 with 12 GB VRAM and approximately 48 GB system RAM. This is sufficient for small or quantized local experiments, but it is not sufficient for reliably running larger open-weight models such as Qwen3 14B/32B, Gemma 3 27B, or Llama 3.1 70B under consistent experimental conditions. Local runs also risk inconsistent offloading between GPU and CPU memory, which would complicate runtime comparisons and hardware-accessibility analysis.

Snellius is therefore appropriate because it provides access to high-memory GPU nodes and a controlled HPC environment. This allows the project to run larger models without uncontrolled CPU offloading and to record compute usage consistently.

## Draft Technical Case: Snellius

### Numerical Methods And Implementation

The project uses autoregressive text generation with pre-trained transformer-based language models. No numerical discretisation, simulation, or model training is performed. The computational workload consists of batched or sequential inference over a fixed prompt bank.

Implementation will use Python, PyTorch, Hugging Face Transformers, Accelerate, and optional BitsAndBytes quantization. Jobs will be submitted through Slurm. Each job will load one model, generate responses for a fixed subset of prompts, and write structured outputs to CSV/XLSX/JSONL files. The workload is embarrassingly parallel across models and prompts, although each individual model run requires GPU memory sufficient to load the selected model.

### Planned Software

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Accelerate
- BitsAndBytes or native reduced precision where appropriate
- pandas / openpyxl for output files
- Slurm batch scripts

### Candidate Models

Local baseline:

- Qwen3 4B Instruct 2507
- Qwen3 8B
- Gemma 3 4B IT
- Gemma 3 12B IT

Snellius extension:

- Qwen3 14B
- Qwen3 32B
- Qwen3 30B-A3B Instruct 2507
- Gemma 3 27B IT
- Llama 3.1 70B Instruct, if approved and feasible

### Data And Storage

Input data consists of a prompt bank of approximately 60 health-related queries in XLSX/CSV form. This is less than 10 MB. Model outputs are text files and spreadsheets. Even with multiple models and repeated generations, the final output data is expected to remain below 5 GB. Temporary model caches may require substantially more storage during execution, potentially 100-500 GB depending on which models are downloaded and retained.

Long-term storage is not requested from SURF. Final outputs will be stored locally and, where appropriate, in the thesis repository or university storage. Downloaded model weights can be removed after the experiment.

## Provisional Compute Estimate

This must be validated with a SURF advisor and, ideally, a small benchmark run.

Assumptions:

- 60 prompts
- 1-4 generations per prompt depending on feasibility
- 350 generated tokens per response
- one GPU per model run
- no training or fine-tuning

The likely resource need is substantially below the NWO large Snellius threshold of 1,000,000 SBU unless the experiment is expanded. A small SURF GPU allocation or pilot allocation may be the appropriate route.

## Draft Email To SURF

Subject: Advice requested on computing-time route for Master's thesis LLM inference study

Dear SURF support team,

I am preparing a Master's thesis project at the University of Amsterdam on the quality of LLM-generated pre-clinical health information. The study involves controlled inference runs with open-weight instruction-tuned language models, comparing model families and parameter sizes such as Qwen3, Gemma 3, and potentially Llama 3.1.

The workload consists only of inference, not model training or fine-tuning. Approximately 60 health-related prompts will be submitted to each model under fixed generation settings. Outputs will be scored using the DISCERN framework for consumer health information quality. Larger models such as Qwen3 14B/32B, Gemma 3 27B, and Llama 3.1 70B exceed the reliable capacity of my local GPU setup.

Could you advise whether this project should be submitted as a small computing-time request through SURF or as a large NWO Rekentijd Nationale Computerfaciliteiten application? I would also appreciate advice on the most appropriate Snellius GPU partition, expected accounting units for this inference-only workload, and whether a short pilot/benchmark allocation is recommended before a full request.

Kind regards,

Jaimy van Hattem

## Information Still Needed

- Name, title, and email of the main applicant / PI.
- Whether the PI has a permanent or sufficiently long temporary contract.
- Supervisor / guarantor information if the main applicant is temporary.
- Exact desired start date and end date.
- Whether this is an individual or group application.
- Whether the project is confidential.
- Preferred route: small SURF request, large NWO request, or ask SURF first.
- Final model list.
- Number of repeats per prompt.
- Whether Llama 3.1 70B is essential or optional.
- Whether Hugging Face gated model access is already approved for Gemma/Llama.
- Desired storage amount for model cache and outputs.
- Whether any SURF expertise hours should be requested.

