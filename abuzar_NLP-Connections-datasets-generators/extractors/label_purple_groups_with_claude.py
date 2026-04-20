import csv
import json
import os
import time
from json import JSONDecodeError

from anthropic import Anthropic

INPUT_PATH = "purple_groups_sample.csv"
OUTPUT_PATH = "purple_groups_labeled.csv"

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
BATCH_SIZE = 25
SLEEP_SECONDS = 1

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


SYSTEM_PROMPT = """
You are labeling candidate PURPLE groups for a Connections-style word puzzle.

Each row has:
- category
- mechanism
- word1, word2, word3, word4
- answer

Your job is to judge whether the group is playable for humans.

Return:
- llm_label: "keep" or "reject"
- llm_score: integer from 1 to 5
- llm_reason: short explanation

Score meaning:
1 = clearly bad
2 = weak
3 = borderline
4 = good
5 = excellent

Important:
Purple groups are usually STRUCTURAL or WORDPLAY-based.
Do NOT require the four displayed words to be semantically similar.
For example, coat, bow, storm, fall -> rain is good even though coat/bow/storm/fall are not semantically close.

General reject rules:
- Reject if the group feels like a technical artifact instead of a fair puzzle.
- Reject if one or more displayed words do not fit the same mechanism.
- Reject if the answer appears directly inside a displayed word.
- Reject if the words are mostly inflections, abbreviations, fragments, or obscure dictionary junk.
- Reject vulgar/offensive/inappropriate words.
- Reject if a human player would probably not infer the intended answer.
- Reject if there is a much better answer than the given answer.

Mechanism rules:

compound_prefix:
The answer should go BEFORE each displayed word to form natural, recognizable compounds.
Example keep:
answer = rain
words = coat, bow, storm, fall
full compounds = raincoat, rainbow, rainstorm, rainfall
Reject if the full words are merely prefix artifacts, morphology, or unnatural strings.

compound_suffix:
The answer should go AFTER each displayed word to form natural, recognizable compounds.
Example keep:
answer = room
words = school, class, bath, bed
full compounds = schoolroom, classroom, bathroom, bedroom
Reject if the result is just a suffix pattern like -ship, -ness, -tion, -able, etc. unless the compounds are truly natural and puzzle-friendly.

anagram:
The four displayed words should be anagrams of each other and reasonably common.
Do NOT require semantic similarity.
Keep if the anagram set is clean and recognizable.
Reject if the words are obscure, archaic, abbreviation-like, or dictionary junk.

verb_noun_association:
The answer should be a verb strongly suggested by the four displayed nouns.
Example keep:
answer = eat
words = meal, lunch, breakfast, dinner
Reject if the nouns suggest a different verb more strongly.
Reject if nouns are too broad, too abstract, or only loosely connected to the answer.

Output only valid JSON in this exact structure:
{
  "labels": [
    {
      "row_id": 0,
      "llm_label": "keep",
      "llm_score": 5,
      "llm_reason": "Short reason."
    }
  ]
}
""".strip()


def load_rows(path):
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def save_rows(rows, path):
    fieldnames = [
        "category",
        "mechanism",
        "word1",
        "word2",
        "word3",
        "word4",
        "answer",
        "llm_label",
        "llm_score",
        "llm_reason",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({
                "category": row.get("category", ""),
                "mechanism": row.get("mechanism", ""),
                "word1": row.get("word1", ""),
                "word2": row.get("word2", ""),
                "word3": row.get("word3", ""),
                "word4": row.get("word4", ""),
                "answer": row.get("answer", ""),
                "llm_label": row.get("llm_label", ""),
                "llm_score": row.get("llm_score", ""),
                "llm_reason": row.get("llm_reason", ""),
            })


def load_existing_or_sample():
    if os.path.exists(OUTPUT_PATH):
        print(f"Resuming from {OUTPUT_PATH}")
        rows = load_rows(OUTPUT_PATH)
    else:
        print(f"Starting from {INPUT_PATH}")
        rows = load_rows(INPUT_PATH)

    for row in rows:
        row.setdefault("llm_label", "")
        row.setdefault("llm_score", "")
        row.setdefault("llm_reason", "")

    return rows


def row_needs_label(row):
    return not row.get("llm_label") or not row.get("llm_score")


def make_unlabeled_batches(rows, batch_size):
    pending = [
        (index, row)
        for index, row in enumerate(rows)
        if row_needs_label(row)
    ]

    for start in range(0, len(pending), batch_size):
        yield pending[start:start + batch_size]


def batch_to_prompt_rows(batch):
    prompt_rows = []

    for row_id, row in batch:
        prompt_rows.append({
            "row_id": row_id,
            "category": row["category"],
            "mechanism": row["mechanism"],
            "word1": row["word1"],
            "word2": row["word2"],
            "word3": row["word3"],
            "word4": row["word4"],
            "answer": row["answer"],
        })

    return prompt_rows


def extract_json_object(text):
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("Could not find JSON object in Claude response")

    return json.loads(text[start:end + 1])


def label_batch(batch):
    prompt_rows = batch_to_prompt_rows(batch)

    user_prompt = {
        "task": "Label these candidate purple groups.",
        "rows": prompt_rows,
    }

    response = client.messages.create(
        model=MODEL,
        max_tokens=4000,
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
                "content": json.dumps(user_prompt, ensure_ascii=False),
            }
        ],
    )

    text = response.content[0].text
    parsed = extract_json_object(text)

    return parsed["labels"]


def validate_label(label):
    row_id = int(label["row_id"])
    llm_label = str(label.get("llm_label", "")).strip().lower()
    llm_score = int(label.get("llm_score", 0))
    llm_reason = str(label.get("llm_reason", "")).strip()

    if llm_label not in {"keep", "reject"}:
        raise ValueError(f"Invalid llm_label: {llm_label}")

    if llm_score < 1 or llm_score > 5:
        raise ValueError(f"Invalid llm_score: {llm_score}")

    if not llm_reason:
        raise ValueError("Missing llm_reason")

    return row_id, llm_label, llm_score, llm_reason


def apply_labels(rows, labels):
    for label in labels:
        row_id, llm_label, llm_score, llm_reason = validate_label(label)

        rows[row_id]["llm_label"] = llm_label
        rows[row_id]["llm_score"] = str(llm_score)
        rows[row_id]["llm_reason"] = llm_reason


def label_all_rows(rows):
    total_pending = sum(1 for row in rows if row_needs_label(row))

    if total_pending == 0:
        print("All rows are already labeled.")
        return rows

    processed = 0

    for batch in make_unlabeled_batches(rows, BATCH_SIZE):
        first_row = batch[0][0] + 1
        last_row = batch[-1][0] + 1

        print(f"Labeling file rows {first_row}-{last_row} ({processed + len(batch)} of {total_pending} pending)")

        try:
            labels = label_batch(batch)
            apply_labels(rows, labels)
            save_rows(rows, OUTPUT_PATH)

        except (JSONDecodeError, KeyError, ValueError) as error:
            print(f"Could not label batch starting at file row {first_row}: {error}")
            print("Progress saved. Stopping.")
            save_rows(rows, OUTPUT_PATH)
            raise

        processed += len(batch)
        time.sleep(SLEEP_SECONDS)

    return rows


def print_summary(rows):
    kept = sum(1 for row in rows if row.get("llm_label") == "keep")
    rejected = sum(1 for row in rows if row.get("llm_label") == "reject")
    unlabeled = sum(1 for row in rows if row_needs_label(row))

    print(f"Saved {len(rows)} rows to {OUTPUT_PATH}")
    print(f"Keep: {kept}")
    print(f"Reject: {rejected}")
    print(f"Unlabeled: {unlabeled}")


def main():
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise ValueError("Missing ANTHROPIC_API_KEY environment variable")

    rows = load_existing_or_sample()
    rows = label_all_rows(rows)
    save_rows(rows, OUTPUT_PATH)
    print_summary(rows)


if __name__ == "__main__":
    main()
