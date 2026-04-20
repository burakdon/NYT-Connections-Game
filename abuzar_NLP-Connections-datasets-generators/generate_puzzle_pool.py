import argparse
import csv
import json
import random
from datetime import datetime, timezone
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

COLOR_ORDER = ["yellow", "green", "blue", "purple"]
PURPLE_MIN_KEEP_PROBABILITY = 0.80

DATASETS = {
    "yellow": "yellow_candidate_groups.csv",
    "green": "green_candidate_groups.csv",
    "blue": "blue_candidate_groups.csv",
    "purple": "candidate_groups_scored.csv",
}

UNCOUNTABLE_TITLES = {
    "art",
    "attire",
    "baseball equipment",
    "bread",
    "electronic equipment",
    "equipment",
    "furniture",
    "game equipment",
    "golf equipment",
    "graphic art",
    "hair",
    "housing",
    "jewelry",
    "merchandise",
    "photographic equipment",
    "plastic art",
    "young",
}

LOWERCASE_TITLE_WORDS = {"a", "an", "and", "as", "at", "by", "for", "in", "of", "on", "or", "the", "to", "with"}


def data_path(filename):
    data_file = DATA_DIR / filename

    if data_file.exists():
        return data_file

    return BASE_DIR / filename


def load_rows(filename):
    path = data_path(filename)

    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def clean_word(word):
    return word.strip().lower()


def row_words(row):
    return [
        clean_word(row["word1"]),
        clean_word(row["word2"]),
        clean_word(row["word3"]),
        clean_word(row["word4"]),
    ]


def smart_title(text):
    words = text.replace("_", " ").strip().split()
    titled_words = []

    for index, word in enumerate(words):
        if index > 0 and word.lower() in LOWERCASE_TITLE_WORDS:
            titled_words.append(word.lower())
        else:
            titled_words.append(word.title())

    return " ".join(titled_words)


def title_case_answer(answer):
    return smart_title(answer)


def to_gerund(verb):
    verb = verb.strip().lower()

    if verb.endswith("ie"):
        return f"{verb[:-2]}ying"

    if verb.endswith("e") and not verb.endswith(("ee", "ye", "oe")):
        return f"{verb[:-1]}ing"

    return f"{verb}ing"


def pluralize_word(word):
    lower_word = word.lower()

    if lower_word.endswith("man") and lower_word not in {"human", "roman", "german"}:
        return f"{word[:-3]}men"

    if lower_word == "medium":
        return "media"

    if lower_word == "trouser":
        return "trousers"

    if lower_word.endswith(("s", "x", "z", "ch", "sh")):
        return f"{word}es"

    if lower_word.endswith("y") and len(lower_word) > 1 and lower_word[-2] not in "aeiou":
        return f"{word[:-1]}ies"

    return f"{word}s"


def pluralize_title(answer):
    normalized_answer = answer.replace("_", " ").strip().lower()

    if normalized_answer in UNCOUNTABLE_TITLES:
        return title_case_answer(answer)

    if " of " in normalized_answer:
        first_word, rest = normalized_answer.split(" ", 1)
        return smart_title(f"{pluralize_word(first_word)} {rest}")

    parts = normalized_answer.split()
    parts[-1] = pluralize_word(parts[-1])

    return smart_title(" ".join(parts))


def make_group_title(row):
    answer = row["answer"].strip()
    mechanism = row["mechanism"]
    category = row["category"]

    if category == "purple":
        if mechanism == "compound_prefix":
            return f"{answer.upper()} ___"

        if mechanism == "compound_suffix":
            return f"___ {answer.upper()}"

        if mechanism == "anagram":
            return "Anagrams"

        if mechanism == "verb_noun_association":
            return f"Associated With {smart_title(to_gerund(answer))}"

        return title_case_answer(mechanism)

    if category == "blue":
        return f"{title_case_answer(answer)} Terms"

    return pluralize_title(answer)


def prepare_group(row):
    return {
        "category": row["category"],
        "mechanism": row["mechanism"],
        "answer": row["answer"],
        "title": make_group_title(row),
        "words": row_words(row),
    }


def load_group_pool(category):
    rows = load_rows(DATASETS[category])
    pool = []

    for row in rows:
        if row.get("category") != category:
            continue

        if category == "purple":
            if row.get("predicted_label") != "keep":
                continue

            keep_probability = float(row.get("keep_probability", 0))

            if keep_probability < PURPLE_MIN_KEEP_PROBABILITY:
                continue

        words = row_words(row)

        if len(set(words)) != 4:
            continue

        pool.append(row)

    if not pool:
        raise ValueError(f"No usable {category} groups found")

    return pool


def load_pools():
    return {
        category: load_group_pool(category)
        for category in COLOR_ORDER
    }


def choose_group(pool, used_words):
    rows = pool[:]
    random.shuffle(rows)

    for row in rows:
        words = row_words(row)

        if any(word in used_words for word in words):
            continue

        used_words.update(words)
        return prepare_group(row)

    return None


def make_one_puzzle(puzzle_id, pools, max_attempts=500):
    for _ in range(max_attempts):
        used_words = set()
        groups = []

        for category in COLOR_ORDER:
            group = choose_group(pools[category], used_words)

            if group is None:
                break

            groups.append(group)

        if len(groups) != 4:
            continue

        display_words = [word for group in groups for word in group["words"]]
        random.shuffle(display_words)

        return {
            "id": puzzle_id,
            "words": display_words,
            "groups": groups,
        }

    raise RuntimeError(f"Could not create puzzle {puzzle_id} without repeated words")


def make_puzzle_pool(count):
    pools = load_pools()
    puzzles = []

    for puzzle_id in range(1, count + 1):
        puzzles.append(make_one_puzzle(puzzle_id, pools))

    return puzzles, pools


def save_puzzle_pool(puzzles, pools, output_path):
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "puzzle_count": len(puzzles),
        "source_counts": {
            category: len(rows)
            for category, rows in pools.items()
        },
        "puzzles": puzzles,
    }

    with path.open("w") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved {len(puzzles)} puzzles to {path}")

    for category in COLOR_ORDER:
        print(f"{category}: source pool {len(pools[category])}")


def main():
    parser = argparse.ArgumentParser(description="Generate a JSON pool of NLP Connections puzzles.")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--output", default=str(DATA_DIR / "puzzle_pool.json"))
    args = parser.parse_args()

    random.seed(args.seed)

    puzzles, pools = make_puzzle_pool(args.count)
    save_puzzle_pool(puzzles, pools, args.output)


if __name__ == "__main__":
    main()
