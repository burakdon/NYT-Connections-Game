from nltk.corpus import wordnet


def synset_short_name(synset):
    return synset.name().split(".")[0]


def get_root_type_from_path(path):
    names = [synset_short_name(synset) for synset in path]

    if "physical_entity" in names:
        return "physical_entity"

    if "abstraction" in names:
        return "abstraction"

    if len(names) > 1:
        return names[1]

    return names[0] if names else "unknown"


def get_categories_from_path(path, min_depth=4):
    categories = []

    for synset in path:
        name = synset_short_name(synset)

        if synset.min_depth() < min_depth:
            continue

        categories.append(name)

    return categories


def describe_noun_senses(word):
    senses = []

    for synset in wordnet.synsets(word, pos=wordnet.NOUN):
        sense_info = {
            "word": word,
            "synset": synset.name(),
            "definition": synset.definition(),
            "paths": [],
        }

        for path in synset.hypernym_paths():
            root_type = get_root_type_from_path(path)
            categories = get_categories_from_path(path)

            sense_info["paths"].append({
                "root_type": root_type,
                "categories": categories,
            })

        senses.append(sense_info)

    return senses


def print_noun_sense_summary(word):
    print("=" * 80)
    print(word)

    for sense in describe_noun_senses(word):
        print("-" * 80)
        print("Synset:", sense["synset"])
        print("Definition:", sense["definition"])

        for path in sense["paths"]:
            print("Root type:", path["root_type"])
            print("Categories:", " -> ".join(path["categories"]))


#print_noun_sense_summary("piano")
#print_noun_sense_summary("bank")
#print_noun_sense_summary("pitch")
print_noun_sense_summary("apple")
