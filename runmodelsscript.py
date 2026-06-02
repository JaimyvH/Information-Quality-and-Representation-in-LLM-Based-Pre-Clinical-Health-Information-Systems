import pandas as pd
import requests
import time

# ========= SETTINGS =========

INPUT_FILE = r"E:\UvA\Master Thesis\AI_Health_Query_Prompt_Bank_filled.xlsx"
OUTPUT_FILE = "results.xlsx"

MODELS = [
    "llama13",
    "llama70b",
    "qwen14b",
	"qwen2.5:72b"
]

OLLAMA_URL = "http://localhost:11434/api/generate"

# ============================

df = pd.read_excel(INPUT_FILE, header=1)

# Create empty columns
for model in MODELS:
    df[model] = ""

for index, row in df.iterrows():

    prompt = str(row["QUERY"])

    print(f"\nPrompt {index+1}/{len(df)}")

    for model in MODELS:

        print(f"Running model: {model}")

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": 150}
        }

        try:
            response = requests.post(OLLAMA_URL, json=payload)

            result = response.json()["response"]

            df.at[index, model] = result

        except Exception as e:
            df.at[index, model] = f"ERROR: {e}"

        time.sleep(0)

# Save results
df.to_excel(OUTPUT_FILE, index=False)

print("\nDone!")
print(f"Saved to: {OUTPUT_FILE}")