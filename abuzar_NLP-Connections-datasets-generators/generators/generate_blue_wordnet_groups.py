import csv
import random
from collections import defaultdict
from nltk.corpus import wordnet
from nltk.stem import PorterStemmer
from better_profanity import profanity
from itertools import combinations


profanity.load_censor_words()
STEMMER = PorterStemmer()

INPUT_WORDLIST = "wordlist.txt"
OUTPUT_PATH = "blue_candidate_groups.csv"

MAX_GROUPS = 500
MAX_BUCKET_SIZE = 400
MAX_NOUN_SENSES_PER_WORD = 7
MIN_HYPERNYM_DEPTH = 7
MAX_HYPONYM_DESCENDANTS = 120
MIN_LOCATION_HYPERNYM_DEPTH = 8
MAX_LOCATION_HYPONYM_DESCENDANTS = 15
GROUPS_PER_BUCKET = 20
MAX_SHARED_WORDS_PER_SAME_ANSWER = 2
RANDOM_SEED = 5

random.seed(RANDOM_SEED)




def remove_profane_words(words):
    """Remove profane/vulgar words before any grouping happens."""

    clean_words = []

    for word in words:
        if profanity.contains_profanity(word):
            continue

        clean_words.append(word)

    return clean_words


def load_wordlist(path=INPUT_WORDLIST):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def is_good_word(word):
    return word.isalpha() and word.islower() and len(word) >= 3


def clean_answer_from_synset(synset):
    return synset.name().split(".")[0].replace("_", " ")


def count_hyponym_descendants(synset):
    return sum(1 for _ in synset.closure(lambda s: s.hyponyms()))

def hypernym_has_abstraction_path(hypernym):
    """Return True if any full hypernym path goes through abstraction."""

    for path in hypernym.hypernym_paths():
        path_names = [synset.name().split(".")[0] for synset in path]

        if "abstraction" in path_names:
            return True

    return False

def answer_overlaps_display_word(answer, words):
    answer_parts = set(answer.split())
    compact_answer = answer.replace(" ", "")

    for word in words:
        if word in answer_parts:
            return True

        if len(compact_answer) >= 4 and compact_answer in word:
            return True

    return False


def word_root(word):
    return STEMMER.stem(word.lower())


def words_have_substring_leak(words):
    """Reject if one displayed word is visibly contained in another."""

    lowered = [word.lower() for word in words]

    for i, word1 in enumerate(lowered):
        for j, word2 in enumerate(lowered):
            if i == j:
                continue

            if len(word1) >= 4 and word1 in word2:
                return True

    return False


def words_share_root(words):
    """Reject if displayed words share the same rough stem/root."""

    roots = [word_root(word) for word in words]
    return len(set(roots)) < len(roots)


def is_valid_blue_display_group(words):
    if len(set(words)) != 4:
        return False

    if words_have_substring_leak(words):
        return False

    if words_share_root(words):
        return False

    return True


def overlaps_existing_group_too_much(words, accepted_groups):
    """Keep rows with the same answer from being near-duplicates."""

    wordset = set(words)

    for accepted_group in accepted_groups:
        if len(wordset & accepted_group) > MAX_SHARED_WORDS_PER_SAME_ANSWER:
            return True

    return False


def is_good_hypernym(hypernym):
    if hypernym.min_depth() < MIN_HYPERNYM_DEPTH:
        return False

    descendant_count = count_hyponym_descendants(hypernym)

    if descendant_count > MAX_HYPONYM_DESCENDANTS:
        return False

    if hypernym.lexname() == "noun.location":
        if hypernym.min_depth() < MIN_LOCATION_HYPERNYM_DEPTH:
            return False

        if descendant_count > MAX_LOCATION_HYPONYM_DESCENDANTS:
            return False

    if hypernym_has_abstraction_path(hypernym):
        return False

    return True



def build_direct_hypernym_buckets(wordlist):
    buckets = defaultdict(set)

    for word in wordlist:
        if not is_good_word(word):
            continue

        noun_synsets = wordnet.synsets(word, pos=wordnet.NOUN)

        if not noun_synsets:
            continue

        for synset in noun_synsets[:MAX_NOUN_SENSES_PER_WORD]:
            for hypernym in synset.hypernyms():
                if not is_good_hypernym(hypernym):
                    continue

                buckets[hypernym].add(word)

    return buckets


def make_candidate_groups(buckets):
    candidates = []
    accepted_groups_by_answer = defaultdict(list)

    bucket_items = list(buckets.items())
    random.shuffle(bucket_items)

    for hypernym, words in bucket_items:
        words = sorted(words)

        if len(words) < 4:
            continue

        if len(words) > MAX_BUCKET_SIZE:
            continue

        answer = clean_answer_from_synset(hypernym)
        accepted_groups = accepted_groups_by_answer[answer]

        combos = list(combinations(words, 4))
        random.shuffle(combos)

        for combo in combos:
            if not is_valid_blue_display_group(combo):
                continue

            if answer_overlaps_display_word(answer, combo):
                continue

            if overlaps_existing_group_too_much(combo, accepted_groups):
                continue

            candidates.append({
                "category": "blue",
                "mechanism": "wordnet_direct_hypernym",
                "word1": combo[0],
                "word2": combo[1],
                "word3": combo[2],
                "word4": combo[3],
                "answer": answer,
            })

            accepted_groups.append(set(combo))

            if len(accepted_groups) >= GROUPS_PER_BUCKET:
                break

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
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(groups)


def main():
    raw_wordlist = load_wordlist()
    wordlist = remove_profane_words(raw_wordlist)

    print(f"Removed {len(raw_wordlist) - len(wordlist)} profane words")

    buckets = build_direct_hypernym_buckets(wordlist)
    groups = make_candidate_groups(buckets)
    save_groups(groups)
    print(f"Saved {len(groups)} blue groups to {OUTPUT_PATH}")



if __name__ == "__main__":
    main()
