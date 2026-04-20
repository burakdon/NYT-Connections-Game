import json
from itertools import combinations
from urllib.parse import urlencode
from urllib.request import urlopen

from gensim.models import KeyedVectors
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

LEMMATIZER = WordNetLemmatizer()


def is_too_close_to_verb(candidate, verb):
    """Reject candidates that are forms or obvious derivations of the verb."""

    verb_lemma = LEMMATIZER.lemmatize(verb, pos="v")
    candidate_noun_lemma = LEMMATIZER.lemmatize(candidate, pos="n")
    candidate_verb_lemma = LEMMATIZER.lemmatize(candidate, pos="v")

    if candidate == verb:
        return True

    if candidate_noun_lemma == verb_lemma:
        return True

    if candidate_verb_lemma == verb_lemma:
        return True

    if candidate.startswith(verb_lemma) or verb_lemma.startswith(candidate):
        return True

    return False


def load_verb_seeds(path="verb_seed.txt"):
    """Load generated verb seeds from disk."""

    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def load_wordlist(path="wordlist.txt"):
    """Load the existing reduced vocabulary from disk."""

    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def pos_counts(word):
    """Count WordNet senses by part of speech."""

    return {
        "n": len(wordnet.synsets(word, pos=wordnet.NOUN)),
        "v": len(wordnet.synsets(word, pos=wordnet.VERB)),
        "a": len(wordnet.synsets(word, pos=wordnet.ADJ)),
        "r": len(wordnet.synsets(word, pos=wordnet.ADV)),
    }


def is_strictly_verb_dominant(word):
    """Return True if WordNet treats the word as more verb-like than noun/adjective-like."""

    counts = pos_counts(word)

    if counts["v"] == 0:
        return False

    if counts["v"] <= counts["n"]:
        return False

    if counts["v"] <= counts["a"]:
        return False

    return True


def is_base_form_verb(word):
    """Return True only for base-form, verb-dominant verbs."""

    if not word.isalpha() or not word.islower():
        return False

    lemma = LEMMATIZER.lemmatize(word, pos="v")

    if lemma != word:
        return False

    if not is_strictly_verb_dominant(word):
        return False

    return True


def is_noun(word):
    """Return True if WordNet recognizes the word as a noun."""

    return bool(wordnet.synsets(word, pos=wordnet.NOUN))


def get_associated_nouns(verb, wordset, max_results=50):
    """Get nouns statistically associated with a verb from Datamuse."""

    params = urlencode({
        "rel_trg": verb,
        "md": "pf",
        "max": max_results * 5,
    })

    url = f"https://api.datamuse.com/words?{params}"

    with urlopen(url) as response:
        results = json.loads(response.read().decode("utf-8"))

    nouns = []

    for item in results:
        candidate = item["word"].lower()
        tags = item.get("tags", [])

        if candidate not in wordset:
            continue

        if "n" not in tags:
            continue

        if not candidate.isalpha():
            continue

        if not is_noun(candidate):
            continue

        if is_too_close_to_verb(candidate, verb):
            continue

        nouns.append(candidate)

        if len(nouns) == max_results:
            break

    return nouns


def average_pairwise_similarity(words, wv):
    """Return average embedding similarity among all word pairs."""

    scores = []

    for a, b in combinations(words, 2):
        try:
            scores.append(wv.similarity(a, b))
        except KeyError:
            return None

    return sum(scores) / len(scores)


def choose_closest_noun_group(noun_candidates, wv, group_size=4):
    """Choose the 4 nouns that are most mutually similar."""

    best_group = None
    best_score = float("-inf")

    for combo in combinations(noun_candidates, group_size):
        score = average_pairwise_similarity(combo, wv)

        if score is None:
            continue

        if score > best_score:
            best_score = score
            best_group = combo

    if best_group is None:
        raise ValueError("Could not find a close noun group")

    return list(best_group), best_score


def extract_verb_noun_associations(
    verb_seeds,
    wordlist,
    wv,
    min_nouns=4,
    max_nouns=50,
    min_score=0.45,
    max_verbs=None,
):
    """Build verb -> closest associated noun group mappings."""

    wordset = set(wordlist)
    associations = {}
    seen_groups = set()
    checked_verbs = 0

    for verb in verb_seeds:
        if not is_base_form_verb(verb):
            continue

        checked_verbs += 1
        if max_verbs is not None and checked_verbs > max_verbs:
            break

        noun_candidates = get_associated_nouns(
            verb=verb,
            wordset=wordset,
            max_results=max_nouns,
        )

        if len(noun_candidates) < min_nouns:
            continue

        try:
            noun_group, score = choose_closest_noun_group(noun_candidates, wv)
        except ValueError:
            continue

        if score < min_score:
            continue

        group_key = tuple(sorted(noun_group))
        if group_key in seen_groups:
            continue

        seen_groups.add(group_key)
        associations[verb] = (noun_group, score)

    return associations


def save_associations(associations, path="verb_noun_associations.txt"):
    """Save verb-noun associations to a TSV file."""

    with open(path, "w") as f:
        for verb, (nouns, score) in sorted(associations.items()):
            noun_text = ",".join(nouns)
            f.write(f"{verb}\t{score:.4f}\t{noun_text}\n")


def main():
    wordlist = load_wordlist()
    verb_seeds = load_verb_seeds()

    wv = KeyedVectors.load_word2vec_format(
        "vectors.bin",
        binary=True,
        unicode_errors="ignore",
    )

    associations = extract_verb_noun_associations(
        verb_seeds=verb_seeds,
        wordlist=wordlist,
        wv=wv,
        min_nouns=4,
        max_nouns=20,
        min_score=0.45,
    )

    save_associations(associations)
    print(f"Saved {len(associations)} verb-noun associations to verb_noun_associations.txt")


if __name__ == "__main__":
    main()
