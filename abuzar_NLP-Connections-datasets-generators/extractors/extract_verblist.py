from collections import Counter

from nltk.corpus import brown, wordnet
from nltk.stem import WordNetLemmatizer

LEMMATIZER = WordNetLemmatizer()

#BAD_VERB_ENDINGS = (
#    "ing", "ed", "tion", "sion", "ness", "ment", "ity", "ly",
#)


def load_wordlist(path="wordlist.txt"):
    """Load the existing reduced vocabulary from disk."""

    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def get_brown_counts():
    """Count lowercase alphabetic Brown corpus words after lemmatization."""

    brown_words = [
        word.lower()
        for word in brown.words()
        if word.isalpha() and word.islower()
    ]

    return Counter(LEMMATIZER.lemmatize(word, pos="v") for word in brown_words)


def get_pos_counts(word):
    """Count how many WordNet synsets exist for each major part of speech."""

    return {
        "n": len(wordnet.synsets(word, pos=wordnet.NOUN)),
        "v": len(wordnet.synsets(word, pos=wordnet.VERB)),
        "a": len(wordnet.synsets(word, pos=wordnet.ADJ)),
        "r": len(wordnet.synsets(word, pos=wordnet.ADV)),
    }


def is_good_verb_seed(word, brown_counts, min_count=3):
    """Return True if a word looks like a good base-form verb seed."""

    if not word.isalpha() or not word.islower():
        return False

    if len(word) < 3:
        return False

    #if word.endswith(BAD_VERB_ENDINGS):
    #    return False

    if LEMMATIZER.lemmatize(word, pos="v") != word:
        return False

    if brown_counts[word] < min_count:
        return False

    pos_counts = get_pos_counts(word)

    if pos_counts["v"] == 0:
        return False

    if pos_counts["v"] <= pos_counts["n"]:
        return False

    if pos_counts["v"] <= pos_counts["a"]:
        return False

    return True


def score_verb_seed(word, brown_counts):
    """Score verbs by frequency and how verb-dominant they are in WordNet."""

    pos_counts = get_pos_counts(word)
    total_synsets = sum(pos_counts.values())
    verb_ratio = pos_counts["v"] / total_synsets if total_synsets else 0

    return brown_counts[word] * verb_ratio


def save_verb_seeds(verb_seeds, path="verb_seed.txt"):
    """Save verb seeds, one per line."""

    with open(path, "w") as f:
        for word in verb_seeds:
            f.write(word + "\n")


def main():
    wordlist = load_wordlist()
    brown_counts = get_brown_counts()

    candidates = [
        word for word in wordlist
        if is_good_verb_seed(word, brown_counts)
    ]

    candidates = sorted(
        candidates,
        key=lambda word: score_verb_seed(word, brown_counts),
        reverse=True,
    )

    save_verb_seeds(candidates)

    print(f"Saved {len(candidates)} verb seeds to verb_seed.txt")


if __name__ == "__main__":
    main()
