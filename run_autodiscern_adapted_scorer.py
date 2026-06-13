import argparse
import json
import random
import re
import time
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


DISCERN_ITEMS = {
    "discern_q1": "Are the aims clear?",
    "discern_q2": "Does the response achieve its aims?",
    "discern_q3": "Is the response relevant to the user's question?",
    "discern_q4": "Is it clear what sources of information were used?",
    "discern_q5": "Is it clear when the information used or reported was produced?",
    "discern_q6": "Is the response balanced and unbiased?",
    "discern_q7": "Does it provide details of additional sources of support and information?",
    "discern_q8": "Does it refer to areas of uncertainty?",
    "discern_q9": "Does it describe how each treatment or self-care option works?",
    "discern_q10": "Does it describe the benefits of each treatment or self-care option?",
    "discern_q11": "Does it describe the risks of each treatment or self-care option?",
    "discern_q12": "Does it describe what would happen if no treatment is used?",
    "discern_q13": "Does it describe how treatment choices affect quality of life?",
    "discern_q14": "Is it clear that there may be more than one possible treatment choice?",
    "discern_q15": "Does it provide support for shared decision-making?",
    "discern_q16": "Overall quality rating as a source of information about treatment choices.",
}

SYSTEM_PROMPT = """You are applying an AutoDiscern-adapted scoring procedure.
AutoDiscern automated DISCERN-style health information quality assessment for web articles.
In this adaptation, the input is an LLM-generated pre-clinical health response rather than HTML website text, and the full 16-item DISCERN framework is scored.

Score only the response text. Do not reward information that is absent. Do not use outside medical knowledge to correct the answer; judge how well the response itself communicates reliable consumer health information.
Return valid JSON only."""


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run an AutoDiscern-adapted LLM judge over DISCERN scoring items."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default="meta-llama/Llama-3.1-70B-Instruct")
    parser.add_argument("--sheet", default="scoring_items")
    parser.add_argument("--max-new-tokens", type=int, default=900)
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def read_input(path, sheet):
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_excel(path, sheet_name=sheet)


def dtype_from_arg(dtype):
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    return "auto"


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_model(model_id, args):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    quantization_config = None
    if args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
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


def get_input_device(model):
    device = getattr(model, "device", None)
    if device is not None:
        return device
    return model.get_input_embeddings().weight.device


def move_inputs_to_device(inputs, device):
    if torch.is_tensor(inputs):
        return {"input_ids": inputs.to(device)}
    if hasattr(inputs, "data"):
        inputs = inputs.data
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in dict(inputs).items()}


def build_user_prompt(row):
    items = "\n".join(f"- {key}: {text}" for key, text in DISCERN_ITEMS.items())
    return f"""Score this response using the full 16-item DISCERN framework.

Scoring scale for every item:
1 = No / serious or extensive shortcomings
2 = Mostly no
3 = Partially / potentially important shortcomings
4 = Mostly yes
5 = Yes / minimal shortcomings

DISCERN items:
{items}

AutoDiscern-style input object:
entity_id: {row.get('blind_id', '')}
categoryName: LLM generated pre-clinical health response
content source: prompt-response pair, not website HTML

User prompt:
{row.get('prompt', '')}

Response content:
{row.get('response', '')}

Return JSON with:
- technical_error: boolean
- discern_q1 through discern_q16: integer 1-5, or null if technical_error is true
- short_rationale: one concise sentence
"""


def build_inputs(tokenizer, user_prompt):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        try:
            return tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                enable_thinking=False,
                return_tensors="pt",
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
            )
    text = f"System:\n{SYSTEM_PROMPT}\n\nUser:\n{user_prompt}\n\nAssistant:\n"
    return tokenizer(text, return_tensors="pt")


def extract_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in model output: {text[:500]}")
    return json.loads(text[start : end + 1])


def normalize_score(value):
    if value is None:
        return None
    try:
        score = int(value)
    except (TypeError, ValueError):
        return None
    if score < 1 or score > 5:
        return None
    return score


def score_row(tokenizer, model, row, args):
    user_prompt = build_user_prompt(row)
    inputs = move_inputs_to_device(build_inputs(tokenizer, user_prompt), get_input_device(model))
    input_length = inputs["input_ids"].shape[-1]

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id,
    }

    started = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_kwargs)
    elapsed = time.perf_counter() - started

    generated = tokenizer.decode(output_ids[0, input_length:], skip_special_tokens=True).strip()
    parsed = extract_json(generated)
    technical_error = bool(parsed.get("technical_error", False))

    result = {
        "judge_raw_json": json.dumps(parsed, ensure_ascii=False),
        "judge_elapsed_seconds": f"{elapsed:.3f}",
        "judge_error": "",
        "technical_error": technical_error,
        "short_rationale": str(parsed.get("short_rationale", "")),
    }
    for key in DISCERN_ITEMS:
        result[key] = None if technical_error else normalize_score(parsed.get(key))
    return result


def has_existing_score(row):
    return all(pd.notna(row.get(key)) and str(row.get(key)).strip() for key in DISCERN_ITEMS)


def save_output(df, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(output, index=False)
    df.to_csv(output.with_suffix(".csv"), index=False, encoding="utf-8-sig")


def main():
    args = parse_args()
    set_seed(args.seed)

    if args.output.exists() and not args.force:
        df = pd.read_excel(args.output)
        print(f"Resuming from existing output: {args.output}")
    else:
        df = read_input(args.input, args.sheet)
        print(f"Loaded scoring items: {args.input}")

    for column in list(DISCERN_ITEMS) + [
        "technical_error",
        "short_rationale",
        "judge_model_id",
        "judge_elapsed_seconds",
        "judge_raw_json",
        "judge_error",
    ]:
        if column not in df.columns:
            df[column] = ""

    if args.limit:
        run_indices = list(df.index[: args.limit])
    else:
        run_indices = list(df.index)

    tokenizer, model = load_model(args.model_id, args)
    completed = 0

    for index in run_indices:
        if not args.force and has_existing_score(df.loc[index]):
            print(f"[skip] row {index + 1}/{len(df)}")
            continue

        print(f"[score] row {index + 1}/{len(df)}")
        try:
            result = score_row(tokenizer, model, df.loc[index], args)
            for key, value in result.items():
                df.at[index, key] = value
            df.at[index, "judge_model_id"] = args.model_id
            print(f"[ok] row {index + 1}")
        except Exception as exc:
            df.at[index, "judge_error"] = repr(exc)
            print(f"[error] row {index + 1}: {exc!r}")

        completed += 1
        if completed >= args.checkpoint_every:
            save_output(df, args.output)
            completed = 0
            print(f"[saved] {args.output}")

    save_output(df, args.output)
    print(f"Done. Saved to {args.output}")


if __name__ == "__main__":
    main()
