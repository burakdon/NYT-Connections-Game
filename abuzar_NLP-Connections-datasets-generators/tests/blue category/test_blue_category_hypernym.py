from nltk.corpus import wordnet


def show_noun_hypernyms(word):
    for synset in wordnet.synsets(word, pos=wordnet.NOUN):
        print("=" * 80)
        print("Synset:", synset.name())
        print("Definition:", synset.definition())
        print("Depth:", synset.min_depth())

        print("\nDirect hypernyms:")
        for hypernym in synset.hypernyms():
            print(
                "-",
                hypernym.name(),
                "|",
                hypernym.lemma_names(),
                "| depth:",
                hypernym.min_depth(),
                "| definition:",
                hypernym.definition(),
            )

        print("\nFull hypernym paths:")
        for path in synset.hypernym_paths():
            print(" -> ".join(s.name().split(".")[0] for s in path))


show_noun_hypernyms("volume")
