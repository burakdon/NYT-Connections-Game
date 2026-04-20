from collections import Counter

from nltk.stem import PorterStemmer, WordNetLemmatizer

LEMMATIZER = WordNetLemmatizer()
STEMMER = PorterStemmer()


def word_root(word):
    """Return a rough root for comparing related word forms."""

    noun_lemma = LEMMATIZER.lemmatize(word, pos="n")
    verb_lemma = LEMMATIZER.lemmatize(word, pos="v")
    shortest = min([word, noun_lemma, verb_lemma], key=len)

    return STEMMER.stem(shortest)


def nouns_share_root(nouns):
    """Return True if two or more displayed nouns share the same root."""

    roots = [word_root(noun) for noun in nouns]
    root_counts = Counter(roots)

    return any(count >= 2 for count in root_counts.values())


def is_too_close_to_verb(candidate, verb):
    """Reject candidates that are forms or obvious derivations of the verb."""

    verb_lemma = LEMMATIZER.lemmatize(verb, pos="v")
    candidate_noun_lemma = LEMMATIZER.lemmatize(candidate, pos="n")
    candidate_verb_lemma = LEMMATIZER.lemmatize(candidate, pos="v")

    verb_stem = STEMMER.stem(verb_lemma)
    candidate_stem = word_root(candidate)

    if candidate == verb:
        return True

    if candidate_noun_lemma == verb_lemma:
        return True

    if candidate_verb_lemma == verb_lemma:
        return True

    if candidate.startswith(verb_lemma) or verb_lemma.startswith(candidate):
        return True

    if candidate_stem == verb_stem:
        return True

    if len(verb_stem) >= 5 and candidate_stem.startswith(verb_stem):
        return True

    if len(candidate_stem) >= 5 and verb_stem.startswith(candidate_stem):
        return True

    return False


def clean_verb_noun_associations(
    input_path="verb_noun_associations.txt",
    output_path="verb_noun_associations_clean.txt",
    min_score=0.45,
):
    cleaned_rows = []
    seen_groups = set()

    with open(input_path, "r") as f:
        for line in f:
            verb, score_text, noun_text = line.strip().split("\t")
            score = float(score_text)
            nouns = noun_text.split(",")

            if score < min_score:
                continue

            if any(is_too_close_to_verb(noun, verb) for noun in nouns):
                continue

            if nouns_share_root(nouns):
                continue

            group_key = tuple(sorted(nouns))
            if group_key in seen_groups:
                continue

            seen_groups.add(group_key)
            cleaned_rows.append((verb, score, nouns))

    with open(output_path, "w") as f:
        for verb, score, nouns in cleaned_rows:
            f.write(f"{verb}\t{score:.4f}\t{','.join(nouns)}\n")

    print(f"Saved {len(cleaned_rows)} cleaned associations to {output_path}")


if __name__ == "__main__":
    clean_verb_noun_associations()
