import argparse
import time
from pathlib import Path

import pandas as pd
import requests


DEFAULT_INPUT_FILE = Path("AI_Health_Query_Prompt_Bank_filled.xlsx")
DEFAULT_OUTPUT_FILE = Path("results.xlsx")
OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_NUM_PREDICT = 350

SYSTEM_PROMPT = """You are answering a pre-clinical consumer health question.
Give a concise, complete answer in plain language.
Use no more than 220 words.
Include:
- a direct answer to the user's question
- important safety cautions or uncertainty
- when to seek professional medical help
Do not provide a diagnosis. Do not end with an unfinished list."""

DEFAULT_MODELS = [
    "llama13",
    "llama70b",
    "qwen14b",
    "qwen2.5:72b",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the thesis prompt bank through local Ollama models."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT_FILE, type=Path)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_FILE, type=Path)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--repeats", default=1, type=int)
    parser.add_argument("--num-predict", default=DEFAULT_NUM_PREDICT, type=int)
    parser.add_argument(
        "--num-ctx",
        default=4096,
        type=int,
        help="Lower values reduce memory use. 4096 is usually enough for this prompt bank.",
    )
    parser.add_argument("--temperature", default=0.7, type=float)
    parser.add_argument("--top-p", default=0.9, type=float)
    parser.add_argument("--min-p", default=0.05, type=float)
    parser.add_argument("--keep-alive", default="45m")
    parser.add_argument("--timeout", default=1800, type=int)
    parser.add_argument("--checkpoint-every", default=5, type=int)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run prompts even when an output cell already contains a response.",
    )
    return parser.parse_args()


def response_column(model, repeat):
    if repeat == 1:
        return model
    return f"{model}_run{repeat}"


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


def generate(prompt, model, args):
    payload = {
        "model": model,
        "system": SYSTEM_PROMPT,
        "prompt": prompt,
        "stream": False,
        "keep_alive": args.keep_alive,
        "options": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "min_p": args.min_p,
            "num_predict": args.num_predict,
            "num_ctx": args.num_ctx,
        },
    }

    started = time.perf_counter()
    response = requests.post(OLLAMA_URL, json=payload, timeout=args.timeout)
    elapsed = time.perf_counter() - started
    response.raise_for_status()
    data = response.json()

    if "response" not in data:
        raise ValueError(f"Ollama response did not contain a response field: {data}")

    return data["response"], elapsed


def main():
    args = parse_args()
    columns = [
        response_column(model, repeat)
        for model in args.models
        for repeat in range(1, args.repeats + 1)
    ]
    df = load_or_create_results(args.input, args.output, columns)

    if "QUERY" not in df.columns:
        raise ValueError("The input sheet must contain a QUERY column.")

    completed_since_checkpoint = 0
    total_tasks = len(df) * len(args.models) * args.repeats
    done_tasks = 0

    print(f"Models: {', '.join(args.models)}")
    print(f"Prompts: {len(df)}")
    print(f"Repeats per prompt/model: {args.repeats}")
    print(f"Maximum generated tokens: {args.num_predict}")
    print(f"Context length: {args.num_ctx}")
    print()

    # Run one model through the whole prompt bank before switching models.
    # This avoids repeatedly loading large models such as Llama 70B.
    for model in args.models:
        print(f"=== Running model: {model} ===")

        for repeat in range(1, args.repeats + 1):
            column = response_column(model, repeat)
            print(f"--- Repeat {repeat}/{args.repeats} -> column: {column} ---")

            for index, row in df.iterrows():
                done_tasks += 1
                prompt = str(row["QUERY"]).strip()

                if not prompt or prompt.lower() == "nan":
                    set_cell(df, index, column, "ERROR: Missing prompt")
                    continue

                if not args.force and has_response(row.get(column)):
                    print(f"[skip] {done_tasks}/{total_tasks} prompt {index + 1}: already done")
                    continue

                print(f"[run] {done_tasks}/{total_tasks} prompt {index + 1}/{len(df)}")

                try:
                    result, elapsed = generate(prompt, model, args)
                    set_cell(df, index, column, result)
                    print(f"[ok] {model} prompt {index + 1} in {elapsed:.1f}s")
                except Exception as exc:
                    set_cell(df, index, column, f"ERROR: {exc}")
                    print(f"[error] {model} prompt {index + 1}: {exc}")

                completed_since_checkpoint += 1
                if completed_since_checkpoint >= args.checkpoint_every:
                    save_results(df, args.output)
                    completed_since_checkpoint = 0
                    print(f"[saved] {args.output}")

        save_results(df, args.output)
        print(f"[saved] completed model {model}")

    save_results(df, args.output)
    print()
    print("Done.")
    print(f"Saved to: {args.output}")
    print(f"CSV backup: {args.output.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
