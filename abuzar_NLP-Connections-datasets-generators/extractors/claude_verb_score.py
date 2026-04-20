import csv
import json
import os
import re
import time
from pathlib import Path

from anthropic import Anthropic

INPUT_PATH = "verb_noun_associations_clean.txt"
OUTPUT_CSV_PATH = "verb_noun_associations_llm_scored2.csv"
KEEP_OUTPUT_PATH = "verb_noun_associations_llm_keep2.txt"

MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")
BATCH_SIZE = 75
SLEEP_SECONDS = 1.0

SYSTEM_PROMPT = """Evaluate Connections-style groups.

Each row is:
verb<TAB>embedding_score<TAB>noun1,noun2,noun3,noun4

The player sees only the nouns. The hidden answer is the verb.

Score:
5 excellent, puzzle-ready
4 usable
3 borderline
2 poor
1 bad
0 invalid

Reject/score low if:
- nouns suggest a better hidden verb than the given verb
- hidden word feels noun-like/name-like/not action-like
- nouns are too broad, abstract, obscure, sensitive, or wrong-sense
- any noun is a direct form/derivation of the verb
- nouns share the same root/morphology
- nouns are merely similar to each other but do not naturally clue the verb

Use one tag:
good, too_direct, better_answer_exists, wrong_sense, too_broad, too_abstract, not_action, shared_root, obscure, sensitive, invalid

Return only JSON list with:
verb, embedding_score, nouns, llm_score, tag, brief_reason, keep

keep=true only if llm_score is 4 or 5.
"""


def load_rows(path):
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

            verb, score_text, nouns_text = parts
            nouns = [noun.strip() for noun in nouns_text.split(",") if noun.strip()]

            if len(nouns) != 4:
                print(f"Skipping line {line_number}, expected 4 nouns: {line}")
                continue

            rows.append({
                "verb": verb,
                "embedding_score": score_text,
                "nouns": nouns,
            })

    return rows


def load_already_scored_verbs(path):
    scored = set()

    if not Path(path).exists():
        return scored

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            scored.add(row["verb"])

    return scored


def format_batch(rows):
    lines = []

    for row in rows:
        noun_text = ",".join(row["nouns"])
        lines.append(f'{row["verb"]}\t{row["embedding_score"]}\t{noun_text}')

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
    user_prompt = f"""Score these rows.

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


def append_scores(path, scored_rows):
    file_exists = Path(path).exists()

    fieldnames = [
        "verb",
        "embedding_score",
        "nouns",
        "llm_score",
        "tag",
        "brief_reason",
        "keep",
    ]

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        for row in scored_rows:
            writer.writerow({
                "verb": row["verb"],
                "embedding_score": row["embedding_score"],
                "nouns": ",".join(row["nouns"]) if isinstance(row["nouns"], list) else row["nouns"],
                "llm_score": row["llm_score"],
                "tag": row["tag"],
                "brief_reason": row["brief_reason"],
                "keep": row["keep"],
            })


def write_keep_file(scored_csv_path, output_path):
    kept_rows = []

    with open(scored_csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            keep_value = str(row["keep"]).lower()

            if keep_value not in {"true", "1", "yes"}:
                continue

            kept_rows.append(row)

    with open(output_path, "w") as f:
        for row in kept_rows:
            f.write(f'{row["verb"]}\t{float(row["embedding_score"]):.4f}\t{row["nouns"]}\n')

    print(f"Saved {len(kept_rows)} kept rows to {output_path}")


def main():
    rows = load_rows(INPUT_PATH)
    already_scored = load_already_scored_verbs(OUTPUT_CSV_PATH)
    rows = [row for row in rows if row["verb"] not in already_scored]

    print(f"Input file: {INPUT_PATH}")
    print(f"Rows left to score: {len(rows)}")
    print(f"Model: {MODEL}")
    print(f"Batch size: {BATCH_SIZE}")

    client = Anthropic()

    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start:start + BATCH_SIZE]
        print(f"Scoring batch {start // BATCH_SIZE + 1}: {len(batch)} rows")

        scored_rows = score_batch(client, batch)
        append_scores(OUTPUT_CSV_PATH, scored_rows)
        time.sleep(SLEEP_SECONDS)

    write_keep_file(OUTPUT_CSV_PATH, KEEP_OUTPUT_PATH)


if __name__ == "__main__":
    main()
