import csv
import random
from collections import defaultdict
from itertools import combinations

from Levenshtein import distance as lev
from nltk.stem import PorterStemmer, WordNetLemmatizer

OUTPUT_PATH = "candidate_groups.csv"
MAX_PER_MECHANISM = 500
RANDOM_SEED = 7

random.seed(RANDOM_SEED)

LEMMATIZER = WordNetLemmatizer()
STEMMER = PorterStemmer()


def word_root(word):
    noun_lemma = LEMMATIZER.lemmatize(word, pos="n")
    verb_lemma = LEMMATIZER.lemmatize(word, pos="v")
    shortest = min([word, noun_lemma, verb_lemma], key=len)

    return STEMMER.stem(shortest)


def has_shared_root(words):
    roots = [word_root(word) for word in words]
    return len(set(roots)) < len(roots)


def has_too_similar_words(words, max_distance=2):
    for word1, word2 in combinations(words, 2):
        if lev(word1, word2) <= max_distance:
            return True

        if word1 in word2 or word2 in word1:
            return True

    return False


def is_valid_display_group(words):
    if len(words) != 4:
        return False

    if len(set(words)) != 4:
        return False

    if has_shared_root(words):
        return False

    if has_too_similar_words(words):
        return False

    return True


def add_group(rows, category, mechanism, words, answer):
    if not is_valid_display_group(words):
        return

    rows.append({
        "category": category,
        "mechanism": mechanism,
        "word1": words[0],
        "word2": words[1],
        "word3": words[2],
        "word4": words[3],
        "answer": answer,
    })


def capped_sample(items, cap):
    items = list(items)
    random.shuffle(items)
    return items[:cap]


def choose_valid_groups_from_bucket(bases, hidden_word, max_groups=None):
    candidates = list(combinations(sorted(set(bases)), 4))
    random.shuffle(candidates)

    valid_groups = []

    for combo in candidates:
        words = list(combo)

        if not is_valid_display_group(words):
            continue

        valid_groups.append((words, hidden_word))

        if max_groups is not None and len(valid_groups) >= max_groups:
            break

    return valid_groups


def add_compound_groups(rows, path="compounds_natural.txt"):
    by_prefix = defaultdict(list)
    by_suffix = defaultdict(list)

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            left, right, full_word = line.split("\t")

            by_prefix[left].append((right, full_word))
            by_suffix[right].append((left, full_word))

    prefix_candidates = []

    for hidden_word, entries in by_prefix.items():
        bases = [base for base, _ in entries]
        prefix_candidates.extend(
            choose_valid_groups_from_bucket(bases, hidden_word)
        )

    suffix_candidates = []

    for hidden_word, entries in by_suffix.items():
        bases = [base for base, _ in entries]
        suffix_candidates.extend(
            choose_valid_groups_from_bucket(bases, hidden_word)
        )

    for words, answer in capped_sample(prefix_candidates, MAX_PER_MECHANISM):
        add_group(
            rows=rows,
            category="purple",
            mechanism="compound_prefix",
            words=words,
            answer=answer,
        )

    for words, answer in capped_sample(suffix_candidates, MAX_PER_MECHANISM):
        add_group(
            rows=rows,
            category="purple",
            mechanism="compound_suffix",
            words=words,
            answer=answer,
        )


def canonical_letters(word):
    return "".join(sorted(word))


def add_anagram_groups(rows, path="anagram_words.txt"):
    buckets = defaultdict(list)

    with open(path, "r") as f:
        for line in f:
            word = line.strip()

            if not word:
                continue

            buckets[canonical_letters(word)].append(word)

    candidates = []

    for words in buckets.values():
        unique_words = sorted(set(words))

        for combo in combinations(unique_words, 4):
            combo = list(combo)

            if len(set(combo)) == 4:
                candidates.append(combo)

    for words in capped_sample(candidates, MAX_PER_MECHANISM):
        rows.append({
            "category": "purple",
            "mechanism": "anagram",
            "word1": words[0],
            "word2": words[1],
            "word3": words[2],
            "word4": words[3],
            "answer": "anagram",
        })


def add_verb_noun_groups(rows, path="verb_noun_associations_llm_keep2.txt"):
    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            verb, _, noun_text = line.split("\t")
            nouns = [noun.strip() for noun in noun_text.split(",") if noun.strip()]

            add_group(
                rows=rows,
                category="purple",
                mechanism="verb_noun_association",
                words=nouns,
                answer=verb,
            )


def save_groups(rows, path=OUTPUT_PATH):
    fieldnames = [
        "category",
        "mechanism",
        "word1",
        "word2",
        "word3",
        "word4",
        "answer",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    rows = []

    add_compound_groups(rows)
    add_anagram_groups(rows)
    add_verb_noun_groups(rows)

    save_groups(rows)

    counts = defaultdict(int)
    for row in rows:
        counts[row["mechanism"]] += 1

    print(f"Saved {len(rows)} groups to {OUTPUT_PATH}")

    for mechanism, count in sorted(counts.items()):
        print(f"{mechanism}: {count}")


if __name__ == "__main__":
    main()
