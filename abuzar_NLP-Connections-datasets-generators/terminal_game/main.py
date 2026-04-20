import csv
import random

from gensim.models import KeyedVectors
from Levenshtein import distance as lev
from nltk.tag import pos_tag
from prettytable import PrettyTable


WORDLIST_PATH = "wordlist.txt"
VECTORS_PATH = "vectors.bin"
YELLOW_GROUPS_PATH = "yellow_candidate_groups.csv"
GREEN_GROUPS_PATH = "green_candidate_groups.csv"
BLUE_GROUPS_PATH = "blue_candidate_groups.csv"
PURPLE_GROUPS_PATH = "candidate_groups_scored.csv"

PURPLE_MIN_KEEP_PROBABILITY = 0.80


def load_wordlist(path=WORDLIST_PATH):
    """Load the base word list used for semantic groups."""

    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def load_vectors(path=VECTORS_PATH):
    """Load pre-computed word2vec embeddings."""

    return KeyedVectors.load_word2vec_format(
        path,
        binary=True,
        unicode_errors="ignore",
    )


def row_words(row):
    """Return the four displayed words from a candidate-group CSV row."""

    return [
        row["word1"].strip().lower(),
        row["word2"].strip().lower(),
        row["word3"].strip().lower(),
        row["word4"].strip().lower(),
    ]


def used_key(word):
    """Normalize a word before checking whether it already appears."""

    return word.strip().lower()


def make_group_record(words, answer, category):
    """Store displayed words plus answer metadata for a group."""

    return {
        "words": words,
        "answer": answer,
        "category": category,
    }


def load_candidate_rows(path, category, predicted_label=None, min_keep_probability=None):
    """Load candidate rows for one category from a CSV file."""

    rows = []

    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)

        for row in reader:
            if row.get("category") != category:
                continue

            if predicted_label is not None and row.get("predicted_label") != predicted_label:
                continue

            if min_keep_probability is not None:
                keep_probability = float(row.get("keep_probability", 0))

                if keep_probability < min_keep_probability:
                    continue

            rows.append(row)

    return rows


def choose_csv_group(path, category, used_words, predicted_label=None, min_keep_probability=None):
    """Choose one CSV-backed group without reusing words already in the puzzle."""

    rows = load_candidate_rows(
        path=path,
        category=category,
        predicted_label=predicted_label,
        min_keep_probability=min_keep_probability,
    )

    random.shuffle(rows)

    for row in rows:
        words = row_words(row)

        if len(set(words)) != 4:
            continue

        if any(used_key(word) in used_words for word in words):
            continue

        used_words.update(used_key(word) for word in words)
        return make_group_record(
            words=words,
            answer=row.get("answer", "").strip(),
            category=category,
        )

    raise ValueError(f"Could not choose a valid {category} group from {path}")


def is_valid_semantic_candidate(seed, word, choices, wordset, used_words):
    """Return True if a word can join the current semantic group."""

    if used_key(word) in used_words:
        return False

    if word not in wordset:
        return False

    if word in choices:
        return False

    if lev(seed, word) <= 4:
        return False

    if seed in word or word in seed:
        return False

    if pos_tag([word])[0][1] != pos_tag([seed])[0][1]:
        return False

    return True


def make_semantic_group(wv, wordlist, used_words, category, max_attempts=500):
    """Generate one semantic group using nearest neighbors from word2vec."""

    wordset = set(wordlist)

    for _ in range(max_attempts):
        seed = random.choice(wordlist).strip().lower()

        if seed in used_words or seed not in wv:
            continue

        choices = [seed]

        try:
            similar_words = wv.most_similar(seed, topn=80, restrict_vocab=16000)
        except KeyError:
            continue

        for word, _ in similar_words:
            word = word.lower().strip()

            if is_valid_semantic_candidate(seed, word, choices, wordset, used_words):
                choices.append(word)

            if len(choices) == 4:
                used_words.update(used_key(word) for word in choices)
                return make_group_record(
                    words=choices,
                    answer=f"semantically related to {seed}",
                    category=category,
                )

    raise ValueError("Could not generate a semantic group")


def make_groups():
    """Generate yellow, green, blue, and purple groups for one puzzle."""

    wv = load_vectors()
    wordlist = load_wordlist()
    used_words = set()

    purple_group = choose_csv_group(
        path=PURPLE_GROUPS_PATH,
        category="purple",
        used_words=used_words,
        predicted_label="keep",
        min_keep_probability=PURPLE_MIN_KEEP_PROBABILITY,
    )

    blue_group = choose_csv_group(
        path=BLUE_GROUPS_PATH,
        category="blue",
        used_words=used_words,
    )

    green_group = choose_csv_group(
        path=GREEN_GROUPS_PATH,
        category="green",
        used_words=used_words,
    )

    yellow_group = choose_csv_group(
        path=YELLOW_GROUPS_PATH,
        category="yellow",
        used_words=used_words,
    )

    return [
        yellow_group,
        green_group,
        blue_group,
        purple_group,
    ]


def colour(item, color):
    """Add chosen colour to given text in terminal output."""

    colour_code = {
        1: "\u001b[33;1m",  # yellow
        2: "\u001b[32;1m",  # green
        3: "\u001b[34;1m",  # blue
        4: "\u001b[35;1m",  # purple
        "g": "\u001b[32m",
        "r": "\u001b[31m",
    }

    return f"{colour_code[color]}{item}\u001b[0m"


def group_words(group):
    return group["words"] if isinstance(group, dict) else group


def group_answer(group):
    return group.get("answer", "") if isinstance(group, dict) else ""


def group_category(group):
    return group.get("category", "") if isinstance(group, dict) else ""


def is_one_away_guess(guess, groups, guessed):
    guess_set = {used_key(word) for word in guess}
    guessed_sets = [
        {used_key(word) for word in group}
        for group in guessed
    ]

    for group in groups:
        group_set = {used_key(word) for word in group}

        if group_set in guessed_sets:
            continue

        if len(guess_set & group_set) == 3:
            return True

    return False


def generate_table(groups, guessed=None):
    """Generate a table of words, coloring groups that were already guessed."""

    if guessed is None:
        guessed = []

    display_groups = [group_words(group)[:] for group in groups]

    for i, group in enumerate(display_groups):
        if group in guessed:
            for j, word in enumerate(group):
                group[j] = colour(word, i + 1)

    words = [word for group in display_groups for word in group]
    random.shuffle(words)

    table = PrettyTable()
    table.header = False
    table.padding_width = 5

    for i in range(4):
        row = []

        for j in range(len(display_groups)):
            word_index = i + 4 * j
            row.append(words[word_index])

        table.add_row(row, divider=True)

    return table


def respond(groups):
    """Prompt user to guess the common connection between words in each group."""

    answer_lookup = {}

    for group in groups:
        sorted_words = sorted(group_words(group))
        answer_lookup[tuple(sorted_words)] = {
            "answer": group_answer(group),
            "category": group_category(group),
        }

    groups = [sorted(group_words(group)) for group in groups]

    correct = 0
    incorrect = 0
    guessed = []

    while correct < 4:
        print(f"Group {correct + 1}:")
        print("Enter one word at a time, pressing enter after each\n")

        common = []

        for _ in range(4):
            response = input()
            common.append(response.strip())

        if sorted(common) in groups:
            correct += 1
            print("\n" + colour("Correct!", "g") + "\n")

            solved_key = tuple(sorted(common))
            solved_info = answer_lookup.get(solved_key, {})
            solved_answer = solved_info.get("answer", "")
            solved_category = solved_info.get("category", "")

            if solved_answer:
                print(f"Answer: {solved_answer}")

            if solved_category:
                print(f"Category: {solved_category}")

            print("")
            guessed.append(sorted(common))
            print(generate_table(groups, guessed=guessed))
            print("")
        else:
            incorrect += 1
            one_away = is_one_away_guess(common, groups, guessed)

            if incorrect == 4:
                print("\nYou have made 4 incorrect guesses. Better luck next time!\n")
                print("The correct answers were:\n")
                print(generate_table(groups, guessed=groups))
                print("")

                restart = input("Would you like to play again? (y/n)\n")

                if restart == "y":
                    play()

                break

            print(
                "\n"
                + colour("Incorrect. ", "r")
                + ("One away. " if one_away else "")
                + f"You have {4 - incorrect} incorrect guesses remaining\n"
            )
            print(generate_table(groups, guessed=guessed))
            print("")

        if correct == 4:
            print("That's all 4 connections, well done!\n")
            restart = input("Would you like to play again? (y/n)\n")

            if restart == "y":
                play()

            break


def play():
    """Play the Connections game."""

    print("\n" + colour("Welcome to an NLP-based Connections Game!", 1) + "\n")
    print(
        "The goal is to make 4 groups of 4 words each, "
        "where each group has a common connection\n"
    )

    groups = make_groups()

    table = generate_table(groups)
    print(table)
    print("")

    respond(groups)


play()
