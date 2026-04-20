from collections import defaultdict

from gensim.models import KeyedVectors
from nltk.corpus import wordnet

from nltk.stem import WordNetLemmatizer

LEMMATIZER = WordNetLemmatizer()


def is_base_form_verb(word):
    """Return True only for base-form verbs, not past tense or -ing forms."""

    if not word.isalpha() or not word.islower():
        return False

    if not wordnet.synsets(word, pos=wordnet.VERB):
        return False
    
    if wordnet.synsets(word, pos=wordnet.NOUN):
        return False
    
    if wordnet.synsets(word, pos=wordnet.ADJ):
        return False

    lemma = LEMMATIZER.lemmatize(word, pos="v")

    return lemma == word

def is_verb(word):
    return bool(wordnet.synsets(word, pos=wordnet.VERB))


def is_noun(word):
    return bool(wordnet.synsets(word, pos=wordnet.NOUN))


def load_wordlist(path="wordlist.txt"):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def find_related_nouns(verb, wv, wordset, topn=100, max_nouns=12):
    related_nouns = []

    try:
        similar_words = wv.most_similar(verb, topn=topn, restrict_vocab=16000)
    except KeyError:
        return related_nouns

    for word, score in similar_words:
        word = word.lower().strip()

        if word not in wordset:
            continue

        if word == verb:
            continue

        if not is_noun(word):
            continue

        if is_verb(word):
            continue

        related_nouns.append((word, score))

        if len(related_nouns) == max_nouns:
            break

    return related_nouns


def extract_verb_noun_associations(
    wordlist,
    wv,
    min_nouns=4,
    max_nouns=12,
    topn=100,
):
    wordset = set(wordlist)
    associations = {}

    for word in wordlist:
        if not is_base_form_verb(word):
            continue


        related_nouns = find_related_nouns(
            verb=word,
            wv=wv,
            wordset=wordset,
            topn=topn,
            max_nouns=max_nouns,
        )

        if len(related_nouns) >= min_nouns:
            associations[word] = related_nouns

    return associations


def save_associations(associations, path="verb_noun_associations.txt"):
    with open(path, "w") as f:
        for verb, nouns in sorted(associations.items()):
            noun_text = ",".join(noun for noun, _ in nouns)
            f.write(f"{verb}\t{noun_text}\n")


def main():
    wordlist = load_wordlist()
    wv = KeyedVectors.load_word2vec_format(
        "vectors.bin",
        binary=True,
        unicode_errors="ignore",
    )

    associations = extract_verb_noun_associations(
        wordlist=wordlist,
        wv=wv,
        min_nouns=4,
        max_nouns=12,
        topn=100,
    )

    save_associations(associations)
    print(f"Saved {len(associations)} verb-noun associations to verb_noun_associations.txt")


if __name__ == "__main__":
    main()
