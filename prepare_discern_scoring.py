import argparse
import hashlib
import re
from pathlib import Path

import pandas as pd


METADATA_COLUMNS = [
    "Query ID",
    "Intent Category",
    "Real-world Share (%)",
    "Topic Cluster",
    "Voice",
    "Specificity",
    "Clinical Urgency",
    "Emotional Tone",
    "Information Type Sought",
    "QUERY",
]

DISCERN_COLUMNS = [f"discern_q{i}" for i in range(1, 17)]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Convert model output spreadsheets to a blinded long-format "
            "DISCERN scoring workbook."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Result .xlsx or .csv files produced by run_transformers_prompts.py.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("discern_scoring_items.xlsx"),
        help="Output Excel workbook.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for blind item order.",
    )
    parser.add_argument(
        "--include-errors",
        action="store_true",
        help="Include technical error responses in the scoring_items sheet.",
    )
    return parser.parse_args()


def read_result_file(path):
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {path}")


def is_response_column(column):
    if column in METADATA_COLUMNS:
        return False
    if column.endswith("_seconds"):
        return False
    return True


def parse_response_column(column):
    match = re.match(r"^(?P<model>.+)_run(?P<run>\d+)$", column)
    if match:
        return match.group("model"), int(match.group("run"))
    return column, 1


def is_error_response(value):
    if pd.isna(value):
        return True
    text = str(value).strip()
    return not text or text.startswith("ERROR:")


def make_blind_id(source_file, query_id, model_label, run):
    raw = f"{source_file}|{query_id}|{model_label}|{run}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def make_long_rows(path, df):
    response_columns = [column for column in df.columns if is_response_column(column)]
    rows = []

    for _, row in df.iterrows():
        query_id = row.get("Query ID", "")
        for column in response_columns:
            model_label, run = parse_response_column(column)
            response = row.get(column, "")
            error = is_error_response(response)
            rows.append(
                {
                    "blind_id": make_blind_id(path.name, query_id, model_label, run),
                    "source_file": path.name,
                    "query_id": query_id,
                    "model_label": model_label,
                    "run": run,
                    "is_technical_error": error,
                    "prompt": row.get("QUERY", ""),
                    "response": "" if pd.isna(response) else str(response),
                    "intent_category": row.get("Intent Category", ""),
                    "topic_cluster": row.get("Topic Cluster", ""),
                    "voice": row.get("Voice", ""),
                    "specificity": row.get("Specificity", ""),
                    "clinical_urgency": row.get("Clinical Urgency", ""),
                    "emotional_tone": row.get("Emotional Tone", ""),
                    "information_type_sought": row.get("Information Type Sought", ""),
                }
            )

    return rows


def make_rubric_sheet():
    rubric = [
        (1, "Are the aims clear?"),
        (2, "Does it achieve its aims?"),
        (3, "Is it relevant?"),
        (4, "Is it clear what sources of information were used?"),
        (5, "Is it clear when the information used or reported was produced?"),
        (6, "Is it balanced and unbiased?"),
        (7, "Does it provide details of additional sources of support and information?"),
        (8, "Does it refer to areas of uncertainty?"),
        (9, "Does it describe how each treatment works?"),
        (10, "Does it describe the benefits of each treatment?"),
        (11, "Does it describe the risks of each treatment?"),
        (12, "Does it describe what would happen if no treatment is used?"),
        (13, "Does it describe how treatment choices affect quality of life?"),
        (14, "Is it clear there may be more than one possible treatment choice?"),
        (15, "Does it provide support for shared decision-making?"),
        (16, "Overall quality rating."),
    ]
    return pd.DataFrame(rubric, columns=["discern_question", "description"])


def main():
    args = parse_args()
    all_rows = []

    for path in args.inputs:
        df = read_result_file(path)
        all_rows.extend(make_long_rows(path, df))

    long_df = pd.DataFrame(all_rows)
    if long_df.empty:
        raise ValueError("No response rows were found.")

    scoring_df = long_df.copy()
    if not args.include_errors:
        scoring_df = scoring_df[~scoring_df["is_technical_error"]].copy()

    scoring_df = scoring_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    for column in DISCERN_COLUMNS:
        scoring_df[column] = ""
    scoring_df["scorer_id"] = ""
    scoring_df["scoring_notes"] = ""

    errors_df = long_df[long_df["is_technical_error"]].copy()
    rubric_df = make_rubric_sheet()
    readme_df = pd.DataFrame(
        [
            {
                "note": (
                    "scoring_items is randomized and includes one row per "
                    "non-error model response. Score each DISCERN question "
                    "from 1 to 5. errors lists technical failures that should "
                    "be regenerated or handled separately, not treated as "
                    "valid health-information answers."
                )
            }
        ]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        scoring_df.to_excel(writer, sheet_name="scoring_items", index=False)
        errors_df.to_excel(writer, sheet_name="errors", index=False)
        rubric_df.to_excel(writer, sheet_name="rubric", index=False)
        readme_df.to_excel(writer, sheet_name="readme", index=False)

    scoring_df.to_csv(args.output.with_suffix(".csv"), index=False, encoding="utf-8-sig")
    print(f"Scoring workbook: {args.output}")
    print(f"Scoring CSV: {args.output.with_suffix('.csv')}")
    print(f"Responses for scoring: {len(scoring_df)}")
    print(f"Technical errors: {len(errors_df)}")


if __name__ == "__main__":
    main()
