import csv
from itertools import combinations

import joblib
import pandas as pd
from gensim.models import KeyedVectors
from nltk.stem import PorterStemmer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

INPUT_PATH = "purple_training_data.csv"
MODEL_PATH = "purple_group_classifier.joblib"
VECTORS_PATH = "vectors.bin"

RANDOM_SEED = 7

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


def make_dataset(rows, wv):
    features = [extract_features(row, wv) for row in rows]
    labels = [int(row["label"]) for row in rows]

    return features, labels


def print_feature_importance(model):
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]

    feature_names = list(preprocessor.get_feature_names_out())
    importances = classifier.feature_importances_

    ranked = sorted(
        zip(feature_names, importances),
        key=lambda item: item[1],
        reverse=True,
    )

    print("\nTop feature importances:")
    for name, importance in ranked[:20]:
        print(f"{name}: {importance:.4f}")


def train_model(features, labels):
    features = pd.DataFrame(features)

    categorical_features = ["mechanism"]
    numeric_features = [
        "answer_length",
        "min_word_length",
        "max_word_length",
        "avg_word_length",
        "max_length_gap",
        "min_pairwise_levenshtein",
        "answer_overlaps_words",
        "words_share_root",
        "has_very_similar_spelling",
        "all_words_same_anagram_signature",
        "avg_compound_length",
        "max_compound_length",
        "min_wv_rank",
        "max_wv_rank",
        "avg_wv_rank",
        "answer_wv_rank",
        "unknown_wv_count",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("numeric", "passthrough", numeric_features),
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_SEED,
        class_weight="balanced",
        max_depth=None,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.25,
        random_state=RANDOM_SEED,
        stratify=labels,
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print(classification_report(
        y_test,
        predictions,
        target_names=["reject", "keep"],
    ))

    print_feature_importance(model)

    return model


def main():
    rows = load_rows()

    print("Loading vectors...")
    wv = load_vectors()

    features, labels = make_dataset(rows, wv)

    positives = sum(labels)
    negatives = len(labels) - positives

    print(f"Loaded {len(rows)} training rows")
    print(f"Positive examples: {positives}")
    print(f"Negative examples: {negatives}")

    model = train_model(features, labels)

    joblib.dump(model, MODEL_PATH)

    print(f"\nSaved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
