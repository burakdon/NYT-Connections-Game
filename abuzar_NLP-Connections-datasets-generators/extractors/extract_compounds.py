from collections import Counter
from nltk.corpus import wordnet

from nltk.corpus import brown
from nltk.stem import WordNetLemmatizer

LEMMATIZER = WordNetLemmatizer()
BLOCKED_COMPOUND_PARTS = {
    "able", "ation", "ed", "er", "est", "ing", "ion", "ity", "less", "ly",
    "ment", "ness", "sion", "tion",
}


def lemmatized_forms(word):
    """Return likely lemmas across common parts of speech."""

    return {LEMMATIZER.lemmatize(word, pos) for pos in ("n", "v", "a", "r")}
from nltk.corpus import wordnet

def is_valid_standalone_word(word):
    if len(word) < 3:
        return False

    if not any(ch in "aeiou" for ch in word):
        return False

    return bool(wordnet.synsets(word))


def is_good_compound_split(left, right, full_word, brown_counts):
    if not is_valid_standalone_word(left) or not is_valid_standalone_word(right):
        return False

    if brown_counts[left] < 3 or brown_counts[right] < 3:
        return False

    if len(left) == 3 and len(right) > 6:
        return False

    if len(right) == 3 and len(left) > 6:
        return False

    if full_word.endswith(("ing", "ed", "tion", "sion", "ness", "ment", "ly")):
        return False

    return True

def is_valid_standalone_word(word):
    """Return True if the word looks like a real standalone English word."""

    if len(word) < 3:
        return False

    if not any(ch in "aeiou" for ch in word):
        return False

    if not wordnet.synsets(word):
        return False

    return True


def load_wordlist(path="wordlist.txt"):
    """Load the existing reduced vocabulary from disk."""

    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def extract_compounds(wordlist, brown_counts):
    """Extract likely compounds whose left and right parts are standalone words."""

    wordset = set(wordlist)
    compounds = []

    for word in sorted(wordset):
        if len(word) < 6:
            continue

        for split_index in range(3, len(word) - 2):
            left = word[:split_index]
            right = word[split_index:]

            if left not in wordset or right not in wordset:
                continue
            if not is_good_compound_split(left, right, word, brown_counts):
                continue

            if left in BLOCKED_COMPOUND_PARTS or right in BLOCKED_COMPOUND_PARTS:
                continue

            if brown_counts[left] < 2 or brown_counts[right] < 2:
                continue

            word_lemmas = lemmatized_forms(word)
            left_lemmas = lemmatized_forms(left)
            right_lemmas = lemmatized_forms(right)

            # Skip candidates that are likely inflectional or derivational variants.
            if (left in word_lemmas or right in word_lemmas or
                    bool(word_lemmas & left_lemmas) or
                    bool(word_lemmas & right_lemmas)):
                continue

            compounds.append((left, right, word))

    return compounds


def save_compounds(compounds, path="compounds.txt"):
    """Save extracted compound candidates to a TSV file."""

    with open(path, "w") as f:
        for left, right, word in compounds:
            f.write(f"{left}\t{right}\t{word}\n")


def main():
    """Build compounds from the existing word list without regenerating vectors."""

    brown_words = [word for word in brown.words() if word.isalpha() and word.islower()]
    brown_counts = Counter(LEMMATIZER.lemmatize(word) for word in brown_words)
    wordlist = load_wordlist()
    compounds = extract_compounds(wordlist, brown_counts)
    save_compounds(compounds)
    print(f"Saved {len(compounds)} compounds to compounds.txt")


if __name__ == "__main__":
    main()