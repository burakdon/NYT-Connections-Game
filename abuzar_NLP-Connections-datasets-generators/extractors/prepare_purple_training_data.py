import csv

INPUT_PATH = "purple_groups_labeled.csv"
OUTPUT_PATH = "purple_training_data.csv"

POSITIVE_MIN_SCORE = 4
NEGATIVE_MAX_SCORE = 2


def load_rows(path=INPUT_PATH):
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def should_use_for_training(row):
    label = row["llm_label"].strip().lower()
    score = int(row["llm_score"])

    if label == "keep" and score >= POSITIVE_MIN_SCORE:
        return True

    if label == "reject" and score <= NEGATIVE_MAX_SCORE:
        return True

    return False


def training_label(row):
    label = row["llm_label"].strip().lower()

    if label == "keep":
        return "1"

    if label == "reject":
        return "0"

    raise ValueError(f"Unknown label: {label}")


def save_training_rows(rows, path=OUTPUT_PATH):
    fieldnames = [
        "category",
        "mechanism",
        "word1",
        "word2",
        "word3",
        "word4",
        "answer",
        "label",
        "llm_label",
        "llm_score",
        "llm_reason",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow({
                "category": row["category"],
                "mechanism": row["mechanism"],
                "word1": row["word1"],
                "word2": row["word2"],
                "word3": row["word3"],
                "word4": row["word4"],
                "answer": row["answer"],
                "label": training_label(row),
                "llm_label": row["llm_label"],
                "llm_score": row["llm_score"],
                "llm_reason": row["llm_reason"],
            })


def main():
    rows = load_rows()
    training_rows = [row for row in rows if should_use_for_training(row)]

    save_training_rows(training_rows)

    positives = sum(1 for row in training_rows if training_label(row) == "1")
    negatives = sum(1 for row in training_rows if training_label(row) == "0")

    print(f"Loaded {len(rows)} labeled rows")
    print(f"Saved {len(training_rows)} clean training rows to {OUTPUT_PATH}")
    print(f"Positive examples: {positives}")
    print(f"Negative examples: {negatives}")


if __name__ == "__main__":
    main()
