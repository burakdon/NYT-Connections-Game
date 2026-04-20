import json
import os
import re
import time
from pathlib import Path

from anthropic import Anthropic

INPUT_PATH = "compounds.txt"

SCORED_OUTPUT_PATH = "compounds_llm_scored.jsonl"
NATURAL_OUTPUT_PATH = "compounds_natural.txt"
PATTERN_ONLY_OUTPUT_PATH = "compounds_pattern_only.txt"
REJECT_OUTPUT_PATH = "compounds_reject.txt"

MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")
BATCH_SIZE = 75
SLEEP_SECONDS = 1.0

SYSTEM_PROMPT = """You are evaluating candidate compound-word splits for a Connections-style word puzzle.

Each row has:
left<TAB>right<TAB>full_word

The question is whether the split is a natural compound that an English speaker would reasonably perceive as:
left + right = full_word

Examples:
- rain<TAB>coat<TAB>raincoat => natural_compound
- door<TAB>bell<TAB>doorbell => natural_compound
- school<TAB>room<TAB>schoolroom => natural_compound
- sub<TAB>stance<TAB>substance => pattern_only
- con<TAB>tent<TAB>content => pattern_only
- pro<TAB>test<TAB>protest => pattern_only

Definitions:
natural_compound = left and right are meaningful standalone parts, and full_word is naturally understood as their combination.
pattern_only = the split is technically a prefix/suffix/string pattern but not a natural compound. It may still be useful for a prefix/suffix pattern category.
reject = obscure, invalid, too unnatural, offensive/sensitive, or not useful for a casual puzzle.

Be strict. Do not mark a split natural_compound just because both parts are dictionary words. The full word must feel like a natural compound.

Return only valid JSON. No markdown. No explanation outside JSON.

Return a list of objects with these keys:
left, right, full_word, label, brief_reason

label must be exactly one of:
natural_compound, pattern_only, reject
"""


def load_compounds(path):
    rows = []

    with open(path, "r") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            parts = line.split("\t")

            if len(parts) != 3:
                print(f"Skipping malformed line {line_number}: {line}")
                continue

            left, right, full_word = parts
            rows.append({
                "left": left,
                "right": right,
                "full_word": full_word,
            })

    return rows


def load_already_scored(path):
    scored = set()

    if not Path(path).exists():
        return scored

    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)
            scored.add((row["left"], row["right"], row["full_word"]))

    return scored


def format_batch(rows):
    lines = []

    for row in rows:
        lines.append(f'{row["left"]}\t{row["right"]}\t{row["full_word"]}')

    return "\n".join(lines)


def extract_json_array(text):
    text = text.strip()

    if text.startswith("[") and text.endswith("]"):
        return json.loads(text)

    match = re.search(r"\[.*\]", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Could not find JSON array in Claude response:\n{text}")

    return json.loads(match.group(0))


def score_batch(client, rows):
    user_prompt = f"""Evaluate these rows.

Rows:
{format_batch(rows)}
"""

    message = client.messages.create(
        model=MODEL,
        max_tokens=8000,
        temperature=0,
        system=[
            {
                "type": "text",
                "text": SYSTEM_PROMPT,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {
                "role": "user",
                "content": user_prompt,
            }
        ],
    )

    response_text = message.content[0].text
    return extract_json_array(response_text)


def append_jsonl(path, rows):
    with open(path, "a") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_split_outputs(scored_path):
    natural_rows = []
    pattern_rows = []
    reject_rows = []

    with open(scored_path, "r") as f:
        for line in f:
            if not line.strip():
                continue

            row = json.loads(line)
            label = row["label"]

            output_line = f'{row["left"]}\t{row["right"]}\t{row["full_word"]}\n'

            if label == "natural_compound":
                natural_rows.append(output_line)
            elif label == "pattern_only":
                pattern_rows.append(output_line)
            elif label == "reject":
                reject_rows.append(output_line)
            else:
                reject_rows.append(output_line)

    with open(NATURAL_OUTPUT_PATH, "w") as f:
        f.writelines(natural_rows)

    with open(PATTERN_ONLY_OUTPUT_PATH, "w") as f:
        f.writelines(pattern_rows)

    with open(REJECT_OUTPUT_PATH, "w") as f:
        f.writelines(reject_rows)

    print(f"Saved {len(natural_rows)} rows to {NATURAL_OUTPUT_PATH}")
    print(f"Saved {len(pattern_rows)} rows to {PATTERN_ONLY_OUTPUT_PATH}")
    print(f"Saved {len(reject_rows)} rows to {REJECT_OUTPUT_PATH}")


def main():
    rows = load_compounds(INPUT_PATH)
    already_scored = load_already_scored(SCORED_OUTPUT_PATH)

    rows = [
        row for row in rows
        if (row["left"], row["right"], row["full_word"]) not in already_scored
    ]

    print(f"Rows left to score: {len(rows)}")
    print(f"Model: {MODEL}")
    print(f"Batch size: {BATCH_SIZE}")

    client = Anthropic()

    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start:start + BATCH_SIZE]
        print(f"Scoring batch {start // BATCH_SIZE + 1}: {len(batch)} rows")

        scored_rows = score_batch(client, batch)
        append_jsonl(SCORED_OUTPUT_PATH, scored_rows)

        time.sleep(SLEEP_SECONDS)

    write_split_outputs(SCORED_OUTPUT_PATH)


if __name__ == "__main__":
    main()
