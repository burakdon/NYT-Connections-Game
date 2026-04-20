import csv
from itertools import combinations

import joblib
import pandas as pd
from gensim.models import KeyedVectors
from nltk.stem import PorterStemmer

INPUT_PATH = "candidate_groups.csv"
MODEL_PATH = "purple_group_classifier.joblib"
OUTPUT_PATH = "candidate_groups_scored.csv"
VECTORS_PATH = "vectors.bin"

CATEGORY = "purple"

KEEP_THRESHOLD = 0.80
REJECT_THRESHOLD = 0.30

STEMMER = PorterStemmer()


def load_rows(path=INPUT_PATH):
    with open(path, "r", newline="") as f:
        return list(csv.DictReader(f))


def load_vectors(path=VECTORS_PATH):
    return KeyedVectors.load_word2vec_format(
        path,
        binary=True,
        unicode_errors="ignore",
    )


def canonical_letters(word):
    return "".join(sorted(word.lower()))


def word_root(word):
    return STEMMER.stem(word.lower())


def answer_overlaps_words(answer, words):
    answer = answer.lower().replace(" ", "")
    return any(answer in word.lower() or word.lower() in answer for word in words)


def words_share_root(words):
    roots = [word_root(word) for word in words]
    return len(set(roots)) < len(roots)


def min_word_length(words):
    return min(len(word) for word in words)


def max_word_length(words):
    return max(len(word) for word in words)


def avg_word_length(words):
    return sum(len(word) for word in words) / len(words)


def max_length_gap(words):
    lengths = [len(word) for word in words]
    return max(lengths) - min(lengths)


def levenshtein_distance(a, b):
    if a == b:
        return 0

    if len(a) < len(b):
        a, b = b, a

    previous_row = list(range(len(b) + 1))

    for i, char_a in enumerate(a, start=1):
        current_row = [i]

        for j, char_b in enumerate(b, start=1):
            insertions = previous_row[j] + 1
            deletions = current_row[j - 1] + 1
            substitutions = previous_row[j - 1] + (char_a != char_b)

            current_row.append(min(insertions, deletions, substitutions))

        previous_row = current_row

    return previous_row[-1]


def min_pairwise_levenshtein(words):
    distances = [
        levenshtein_distance(a.lower(), b.lower())
        for a, b in combinations(words, 2)
    ]

    return min(distances)


def has_very_similar_spelling(words):
    for a, b in combinations(words, 2):
        a = a.lower()
        b = b.lower()

        if a in b or b in a:
            return True

        if levenshtein_distance(a, b) <= 2:
            return True

    return False


def all_words_same_anagram_signature(words):
    signatures = [canonical_letters(word) for word in words]
    return len(set(signatures)) == 1


def reconstructed_compound_lengths(mechanism, words, answer):
    if mechanism == "compound_prefix":
        return [len(answer + word) for word in words]

    if mechanism == "compound_suffix":
        return [len(word + answer) for word in words]

    return [0, 0, 0, 0]


def wv_rank(word, wv):
    missing_rank = len(wv.key_to_index) + 1
    return wv.key_to_index.get(word.lower(), missing_rank)


def wv_rank_features(words, answer, wv):
    ranks = [wv_rank(word, wv) for word in words]
    answer_rank = wv_rank(answer, wv)
    missing_rank = len(wv.key_to_index) + 1

    return {
        "min_wv_rank": min(ranks),
        "max_wv_rank": max(ranks),
        "avg_wv_rank": sum(ranks) / len(ranks),
        "answer_wv_rank": answer_rank,
        "unknown_wv_count": sum(1 for rank in ranks if rank == missing_rank),
    }


def extract_features(row, wv):
    mechanism = row["mechanism"]
    answer = row["answer"]
    words = [
        row["word1"],
        row["word2"],
        row["word3"],
        row["word4"],
    ]

    compound_lengths = reconstructed_compound_lengths(mechanism, words, answer)
    rank_features = wv_rank_features(words, answer, wv)

    return {
        "mechanism": mechanism,
        "answer_length": len(answer.replace(" ", "")),
        "min_word_length": min_word_length(words),
        "max_word_length": max_word_length(words),
        "avg_word_length": avg_word_length(words),
        "max_length_gap": max_length_gap(words),
        "min_pairwise_levenshtein": min_pairwise_levenshtein(words),
        "answer_overlaps_words": int(answer_overlaps_words(answer, words)),
        "words_share_root": int(words_share_root(words)),
        "has_very_similar_spelling": int(has_very_similar_spelling(words)),
        "all_words_same_anagram_signature": int(all_words_same_anagram_signature(words)),
        "avg_compound_length": sum(compound_lengths) / len(compound_lengths),
        "max_compound_length": max(compound_lengths),
        "min_wv_rank": rank_features["min_wv_rank"],
        "max_wv_rank": rank_features["max_wv_rank"],
        "avg_wv_rank": rank_features["avg_wv_rank"],
        "answer_wv_rank": rank_features["answer_wv_rank"],
        "unknown_wv_count": rank_features["unknown_wv_count"],
    }


def score_rows(rows, model, wv):
    purple_rows = [row for row in rows if row.get("category") == CATEGORY]

    features = [extract_features(row, wv) for row in purple_rows]
    features = pd.DataFrame(features)

    keep_probabilities = model.predict_proba(features)[:, 1]

    scored_rows = []

    for row, keep_probability in zip(purple_rows, keep_probabilities):
        new_row = dict(row)
        new_row["keep_probability"] = f"{keep_probability:.4f}"

        if keep_probability >= KEEP_THRESHOLD:
            new_row["predicted_label"] = "keep"
        elif keep_probability <= REJECT_THRESHOLD:
            new_row["predicted_label"] = "reject"
        else:
            new_row["predicted_label"] = "borderline"

        scored_rows.append(new_row)

    scored_rows.sort(
        key=lambda row: float(row["keep_probability"]),
        reverse=True,
    )

    return scored_rows


def save_rows(rows, path=OUTPUT_PATH):
    fieldnames = [
        "category",
        "mechanism",
        "word1",
        "word2",
        "word3",
        "word4",
        "answer",
        "predicted_label",
        "keep_probability",
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
                "predicted_label": row.get("predicted_label", ""),
                "keep_probability": row.get("keep_probability", ""),
            })


def print_summary(rows):
    keep = sum(1 for row in rows if row["predicted_label"] == "keep")
    reject = sum(1 for row in rows if row["predicted_label"] == "reject")
    borderline = sum(1 for row in rows if row["predicted_label"] == "borderline")

    print(f"Saved {len(rows)} scored purple groups to {OUTPUT_PATH}")
    print(f"Predicted keep: {keep}")
    print(f"Predicted reject: {reject}")
    print(f"Borderline: {borderline}")


def main():
    rows = load_rows()
    model = joblib.load(MODEL_PATH)

    print("Loading vectors...")
    wv = load_vectors()

    scored_rows = score_rows(rows, model, wv)

    save_rows(scored_rows)
    print_summary(scored_rows)


if __name__ == "__main__":
    main()
