# DISCERN Judge Prompt Template

Use this prompt only for an automated auxiliary scoring workflow. It should not replace expert scoring unless validated against expert scores.

```text
You are scoring a pre-clinical consumer health information response using the DISCERN instrument.

Score each item from 1 to 5:
1 = No / serious shortcomings
3 = Partially / potentially important shortcomings
5 = Yes / minimal shortcomings

Use only the response text. Do not reward the response for information that is absent. If the response contains a technical error, traceback, or empty content, mark all scores as null and set technical_error=true.

Prompt asked by user:
{prompt}

Model response:
{response}

Return valid JSON only with this structure:
{
  "technical_error": false,
  "discern_q1": 1,
  "discern_q2": 1,
  "discern_q3": 1,
  "discern_q4": 1,
  "discern_q5": 1,
  "discern_q6": 1,
  "discern_q7": 1,
  "discern_q8": 1,
  "discern_q9": 1,
  "discern_q10": 1,
  "discern_q11": 1,
  "discern_q12": 1,
  "discern_q13": 1,
  "discern_q14": 1,
  "discern_q15": 1,
  "discern_q16": 1,
  "short_rationale": "One concise sentence explaining the scoring pattern."
}
```

