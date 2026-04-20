import csv
import random
from collections import defaultdict
from itertools import combinations

from better_profanity import profanity
from gensim.models import KeyedVectors
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer


profanity.load_censor_words()
STEMMER = PorterStemmer()

INPUT_WORDLIST = "wordlist.txt"
VECTORS_PATH = "vectors.bin"
OUTPUT_PATH = "green_candidate_groups.csv"

RANDOM_SEED = 11
MAX_GROUPS = 8000
GROUPS_PER_BUCKET = 40
MAX_RANDOM_ATTEMPTS_PER_BUCKET = 1000
MAX_SHARED_WORDS_PER_SAME_ANSWER = 2

MIN_HYPERNYM_DEPTH = 6
MAX_HYPERNYM_DEPTH = 9
MIN_BUCKET_SIZE = 4
MAX_BUCKET_SIZE = 130
MAX_HYPONYM_DESCENDANTS = 450

MAX_DISPLAY_WORD_RANK = 100000
MIN_AVG_PAIRWISE_SIMILARITY = 0.27
MIN_PAIRWISE_SIMILARITY = 0.08
ALLOWED_HYPERNYM_LEXNAMES = {
    "noun.act",
    "noun.animal",
    "noun.artifact",
    "noun.body",
    "noun.communication",
    "noun.food",
    "noun.group",
    "noun.location",
    "noun.person",
    "noun.plant",
}

random.seed(RANDOM_SEED)


def load_wordlist(path=INPUT_WORDLIST):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def load_vectors(path=VECTORS_PATH):
    return KeyedVectors.load_word2vec_format(
        path,
        binary=True,
        unicode_errors="ignore",
    )


def remove_profane_words(words):
    clean_words = []

    for word in words:
        if profanity.contains_profanity(word):
            continue

        clean_words.append(word)

    return clean_words


def is_good_word(word, wv):
    if not word.isalpha() or not word.islower():
        return False

    if len(word) < 3:
        return False

    if word not in wv.key_to_index:
        return False

    if wv.key_to_index[word] > MAX_DISPLAY_WORD_RANK:
        return False

    return True


def clean_answer_from_synset(synset):
    return synset.name().split(".")[0].replace("_", " ")


def answer_is_simple(answer):
    parts = answer.split()

    if len(parts) > 2:
        return False

    return all(part.isalpha() and len(part) >= 3 for part in parts)


def count_hyponym_descendants(synset):
    return sum(1 for _ in synset.closure(lambda s: s.hyponyms()))


def hypernym_has_abstraction_path(hypernym):
    for path in hypernym.hypernym_paths():
        path_names = [synset.name().split(".")[0] for synset in path]

        if "abstraction" in path_names:
            return True

    return False


def is_good_green_hypernym(hypernym):
    if hypernym.lexname() not in ALLOWED_HYPERNYM_LEXNAMES:
        return False

    depth = hypernym.min_depth()

    if depth < MIN_HYPERNYM_DEPTH:
        return False

    if depth > MAX_HYPERNYM_DEPTH:
        return False

    if hypernym_has_abstraction_path(hypernym):
        return False

    if count_hyponym_descendants(hypernym) > MAX_HYPONYM_DESCENDANTS:
        return False

    answer = clean_answer_from_synset(hypernym)

    if not answer_is_simple(answer):
        return False

    return True


def answer_parts_are_known(answer, wv):
    return all(part in wv.key_to_index for part in answer.split())


def answer_matches_primary_noun_sense(answer, hypernym):
    query = answer.replace(" ", "_")
    noun_synsets = wordnet.synsets(query, pos=wordnet.NOUN)

    if not noun_synsets:
        return False

    return noun_synsets[0] == hypernym


def is_playable_green_answer(answer, hypernym, wv):
    if not answer_parts_are_known(answer, wv):
        return False

    if not answer_matches_primary_noun_sense(answer, hypernym):
        return False

    return True


def word_root(word):
    return STEMMER.stem(word.lower())


def words_have_substring_leak(words):
    lowered = [word.lower() for word in words]

    for i, word1 in enumerate(lowered):
        for j, word2 in enumerate(lowered):
            if i == j:
                continue

            if len(word1) >= 4 and word1 in word2:
                return True

    return False


def words_share_root(words):
    roots = [word_root(word) for word in words]
    return len(set(roots)) < len(roots)


def answer_overlaps_display_word(answer, words):
    answer_parts = set(answer.split())

    for word in words:
        if word in answer_parts:
            return True

    return False


def pairwise_similarity_scores(words, wv):
    scores = []

    for word1, word2 in combinations(words, 2):
        if word1 not in wv.key_to_index or word2 not in wv.key_to_index:
            return None

        scores.append(wv.similarity(word1, word2))

    return scores


def semantic_coherence(words, wv):
    scores = pairwise_similarity_scores(words, wv)

    if scores is None:
        return None, None

    return sum(scores) / len(scores), min(scores)


def is_valid_green_display_group(words, answer, wv):
    if len(set(words)) != 4:
        return False, None

    if answer_overlaps_display_word(answer, words):
        return False, None

    if words_have_substring_leak(words):
        return False, None

    if words_share_root(words):
        return False, None

    avg_similarity, min_similarity = semantic_coherence(words, wv)

    if avg_similarity is None:
        return False, None

    if avg_similarity < MIN_AVG_PAIRWISE_SIMILARITY:
        return False, None

    if min_similarity < MIN_PAIRWISE_SIMILARITY:
        return False, None

    return True, avg_similarity


def overlaps_existing_group_too_much(words, accepted_groups):
    """Keep rows with the same answer from being near-duplicates."""

    wordset = set(words)

    for accepted_group in accepted_groups:
        if len(wordset & accepted_group) > MAX_SHARED_WORDS_PER_SAME_ANSWER:
            return True

    return False


def hypernym_ancestors_for_synset(synset):
    ancestors = set()

    for path in synset.hypernym_paths():
        for hypernym in path[:-1]:
            if is_good_green_hypernym(hypernym):
                ancestors.add(hypernym)

    return ancestors


def build_green_hypernym_buckets(wordlist, wv):
    buckets = defaultdict(set)

    for word in wordlist:
        if not is_good_word(word, wv):
            continue

        for synset in wordnet.synsets(word, pos=wordnet.NOUN):
            for hypernym in hypernym_ancestors_for_synset(synset):
                buckets[hypernym].add(word)

    return buckets


def candidate_combos_from_bucket(words):
    possible_combo_count = len(words) * (len(words) - 1) * (len(words) - 2) * (len(words) - 3) // 24

    if possible_combo_count <= MAX_RANDOM_ATTEMPTS_PER_BUCKET:
        combos = list(combinations(words, 4))
        random.shuffle(combos)
        return combos

    combos = []
    seen = set()

    for _ in range(MAX_RANDOM_ATTEMPTS_PER_BUCKET):
        combo = tuple(sorted(random.sample(words, 4)))

        if combo in seen:
            continue

        seen.add(combo)
        combos.append(combo)

    return combos


def make_candidate_groups(buckets, wv):
    candidates = []
    accepted_groups_by_answer = defaultdict(list)

    bucket_items = list(buckets.items())
    random.shuffle(bucket_items)

    for hypernym, words in bucket_items:
        words = sorted(words)

        if len(words) < MIN_BUCKET_SIZE:
            continue

        if len(words) > MAX_BUCKET_SIZE:
            continue

        answer = clean_answer_from_synset(hypernym)

        if not is_playable_green_answer(answer, hypernym, wv):
            continue

        bucket_count = 0
        accepted_groups = accepted_groups_by_answer[answer]

        for combo in candidate_combos_from_bucket(words):
            combo = list(combo)
            is_valid, coherence_score = is_valid_green_display_group(combo, answer, wv)

            if not is_valid:
                continue

            if overlaps_existing_group_too_much(combo, accepted_groups):
                continue

            candidates.append({
                "category": "green",
                "mechanism": "wordnet_common_hypernym",
                "word1": combo[0],
                "word2": combo[1],
                "word3": combo[2],
                "word4": combo[3],
                "answer": answer,
                "coherence_score": f"{coherence_score:.4f}",
                "hypernym_depth": hypernym.min_depth(),
                "bucket_size": len(words),
            })

            bucket_count += 1
            accepted_groups.append(set(combo))

            if bucket_count >= GROUPS_PER_BUCKET:
                break

            if len(candidates) >= MAX_GROUPS:
                random.shuffle(candidates)
                return candidates

    random.shuffle(candidates)
    return candidates


def save_groups(groups, path=OUTPUT_PATH):
    fieldnames = [
        "category",
        "mechanism",
        "word1",
        "word2",
        "word3",
        "word4",
        "answer",
        "coherence_score",
        "hypernym_depth",
        "bucket_size",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(groups)


def main():
    raw_wordlist = load_wordlist()
    wordlist = remove_profane_words(raw_wordlist)

    print(f"Removed {len(raw_wordlist) - len(wordlist)} profane words")
    print("Loading vectors...")
    wv = load_vectors()

    print("Building green WordNet buckets...")
    buckets = build_green_hypernym_buckets(wordlist, wv)

    print("Generating green candidate groups...")
    groups = make_candidate_groups(buckets, wv)
    save_groups(groups)

    print(f"Saved {len(groups)} green groups to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
