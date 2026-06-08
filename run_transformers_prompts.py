import argparse
import gc
import json
import random
import time
import traceback
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


DEFAULT_INPUT_FILE = Path("AI_Health_Query_Prompt_Bank_filled.xlsx")
DEFAULT_OUTPUT_FILE = Path("results_transformers.xlsx")
DEFAULT_MODEL_CONFIG = Path("model_configs.json")
DEFAULT_NUM_PREDICT = 350

SYSTEM_PROMPT = """You are answering a pre-clinical consumer health question.
Give a concise, complete answer in plain language.
Use no more than 220 words.
Include:
- a direct answer to the user's question
- important safety cautions or uncertainty
- when to seek professional medical help
Do not provide a diagnosis. Do not end with an unfinished list."""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the thesis prompt bank with Hugging Face Transformers."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT_FILE, type=Path)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_FILE, type=Path)
    parser.add_argument("--model-config", default=DEFAULT_MODEL_CONFIG, type=Path)
    parser.add_argument("--profile", default="local_qwen_scaling")
    parser.add_argument(
        "--models",
        nargs="+",
        help="Optional explicit Hugging Face model IDs. Overrides --profile.",
    )
    parser.add_argument("--repeats", default=1, type=int)
    parser.add_argument("--max-new-tokens", default=DEFAULT_NUM_PREDICT, type=int)
    parser.add_argument("--temperature", default=0.7, type=float)
    parser.add_argument("--top-p", default=0.9, type=float)
    parser.add_argument("--min-p", default=0.05, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16"])
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable Qwen3 thinking mode when the tokenizer supports it. Disabled by default.",
    )
    parser.add_argument("--checkpoint-every", default=5, type=int)
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately after a generation error. Useful for debugging.",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def load_model_entries(args):
    if args.models:
        return [
            {"label": model_id.replace("/", "__").replace("-", "_"), "model_id": model_id}
            for model_id in args.models
        ]

    with args.model_config.open("r", encoding="utf-8") as file:
        profiles = json.load(file)

    if args.profile not in profiles:
        available = ", ".join(sorted(profiles))
        raise ValueError(f"Unknown profile '{args.profile}'. Available profiles: {available}")

    return profiles[args.profile]


def response_column(label, repeat):
    if repeat == 1:
        return label
    return f"{label}_run{repeat}"


def seconds_column(label, repeat):
    return f"{response_column(label, repeat)}_seconds"


def load_or_create_results(input_file, output_file, columns):
    if output_file.exists():
        df = pd.read_excel(output_file)
        print(f"Resuming from existing output file: {output_file}")
    else:
        df = pd.read_excel(input_file, header=1)
        print(f"Loaded prompt bank: {input_file}")

    for column in columns:
        if column not in df.columns:
            df[column] = ""
        df[column] = df[column].astype("object")

    return df


def has_response(value):
    if pd.isna(value):
        return False
    text = str(value).strip()
    return bool(text) and not text.startswith("ERROR:")


def set_cell(df, index, column, value):
    if df[column].dtype != "object":
        df[column] = df[column].astype("object")
    df.at[index, column] = str(value)


def save_results(df, output_file):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_file, index=False)
    df.to_csv(output_file.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def dtype_from_arg(dtype):
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    return "auto"


def load_tokenizer_and_model(model_id, args):
    print(f"Loading model: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype_from_arg(args.dtype),
        device_map="auto",
        quantization_config=quantization_config,
        trust_remote_code=True,
    )
    model.eval()
    return tokenizer, model


def build_inputs(tokenizer, prompt, enable_thinking):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        try:
            return tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
                return_tensors="pt",
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )

    fallback_prompt = (
        f"System:\n{SYSTEM_PROMPT}\n\n"
        f"User:\n{prompt}\n\n"
        "Assistant:\n"
    )
    return tokenizer(fallback_prompt, return_tensors="pt").input_ids


def generate_response(tokenizer, model, prompt, args, seed):
    set_seed(seed)
    input_ids = build_inputs(tokenizer, prompt, args.enable_thinking).to(model.device)

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "pad_token_id": tokenizer.eos_token_id,
    }

    # min_p is available in recent Transformers versions.
    if args.min_p is not None:
        generation_kwargs["min_p"] = args.min_p

    started = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(input_ids, **generation_kwargs)
    elapsed = time.perf_counter() - started

    new_tokens = output_ids[0, input_ids.shape[-1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return text, elapsed


def unload_model(model, tokenizer):
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    model_entries = load_model_entries(args)
    columns = []
    for entry in model_entries:
        for repeat in range(1, args.repeats + 1):
            columns.append(response_column(entry["label"], repeat))
            columns.append(seconds_column(entry["label"], repeat))

    df = load_or_create_results(args.input, args.output, columns)
    if "QUERY" not in df.columns:
        raise ValueError("The input sheet must contain a QUERY column.")

    print(f"Profile: {args.profile}")
    print(f"Models: {', '.join(entry['label'] for entry in model_entries)}")
    print(f"Prompts: {len(df)}")
    print(f"Repeats: {args.repeats}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Seed base: {args.seed}")
    print()

    completed_since_checkpoint = 0

    for entry in model_entries:
        label = entry["label"]
        model_id = entry["model_id"]
        tokenizer, model = load_tokenizer_and_model(model_id, args)

        try:
            for repeat in range(1, args.repeats + 1):
                response_col = response_column(label, repeat)
                elapsed_col = seconds_column(label, repeat)
                print(f"=== {label} repeat {repeat}/{args.repeats} ===")

                for index, row in df.iterrows():
                    prompt = str(row["QUERY"]).strip()
                    if not prompt or prompt.lower() == "nan":
                        set_cell(df, index, response_col, "ERROR: Missing prompt")
                        continue

                    if not args.force and has_response(row.get(response_col)):
                        print(f"[skip] {label} repeat {repeat} prompt {index + 1}")
                        continue

                    print(f"[run] {label} repeat {repeat} prompt {index + 1}/{len(df)}")
                    try:
                        run_seed = args.seed + repeat - 1
                        result, elapsed = generate_response(
                            tokenizer, model, prompt, args, run_seed
                        )
                        set_cell(df, index, response_col, result)
                        set_cell(df, index, elapsed_col, f"{elapsed:.3f}")
                        print(f"[ok] {label} prompt {index + 1} in {elapsed:.1f}s")
                    except Exception as exc:
                        error_text = "".join(
                            traceback.format_exception_only(type(exc), exc)
                        ).strip()
                        if not error_text:
                            error_text = repr(exc)
                        set_cell(df, index, response_col, f"ERROR: {error_text}")
                        print(f"[error] {label} prompt {index + 1}: {error_text}")
                        if args.stop_on_error:
                            raise

                    completed_since_checkpoint += 1
                    if completed_since_checkpoint >= args.checkpoint_every:
                        save_results(df, args.output)
                        completed_since_checkpoint = 0
                        print(f"[saved] {args.output}")

            save_results(df, args.output)
            print(f"[saved] completed model {label}")
        finally:
            unload_model(model, tokenizer)

    save_results(df, args.output)
    print("Done.")
    print(f"Saved to: {args.output}")
    print(f"CSV backup: {args.output.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
