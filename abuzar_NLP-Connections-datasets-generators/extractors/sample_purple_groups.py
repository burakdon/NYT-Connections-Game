import csv
import random
from collections import defaultdict

INPUT_PATH = "candidate_groups.csv"
OUTPUT_PATH = "purple_groups_sample.csv"

CATEGORY = "purple"
SAMPLES_PER_MECHANISM = 100
RANDOM_SEED = 7

random.seed(RANDOM_SEED)


def load_candidate_groups(path=INPUT_PATH):
    """Load candidate groups from CSV."""

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def group_rows_by_mechanism(rows, category=CATEGORY):
    """Keep one category and group rows by mechanism."""

    grouped = defaultdict(list)

    for row in rows:
        if row["category"] != category:
            continue

        mechanism = row["mechanism"]
        grouped[mechanism].append(row)

    return grouped


def sample_rows_by_mechanism(grouped_rows, samples_per_mechanism=SAMPLES_PER_MECHANISM):
    """Randomly sample the same number of rows from each mechanism."""

    sampled_rows = []

    for mechanism, rows in sorted(grouped_rows.items()):
        rows = rows[:]
        random.shuffle(rows)

        sample_size = min(samples_per_mechanism, len(rows))
        sampled = rows[:sample_size]

        print(f"{mechanism}: sampled {sample_size} out of {len(rows)}")

        sampled_rows.extend(sampled)

    random.shuffle(sampled_rows)
    return sampled_rows


def add_empty_llm_columns(rows):
    """Add blank columns that can be filled by an LLM labeler later."""

    updated_rows = []

    for row in rows:
        new_row = dict(row)
        new_row["llm_label"] = ""
        new_row["llm_score"] = ""
        new_row["llm_reason"] = ""
        updated_rows.append(new_row)

    return updated_rows


def save_rows(rows, path=OUTPUT_PATH):
    """Save sampled rows to CSV."""

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
        writer.writerows(rows)


def main():
    rows = load_candidate_groups()
    grouped_rows = group_rows_by_mechanism(rows)
    sampled_rows = sample_rows_by_mechanism(grouped_rows)
    sampled_rows = add_empty_llm_columns(sampled_rows)

    save_rows(sampled_rows)

    print(f"Saved {len(sampled_rows)} sampled purple groups to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
