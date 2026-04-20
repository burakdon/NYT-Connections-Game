from pathlib import Path

NATURAL_PATH = Path("compounds_natural.txt")
PATTERN_PATH = Path("compounds_pattern_only.txt")
#BACKUP_PATH = Path("compounds_natural.before_suffix_cleanup.txt")

SUFFIX_LIKE_PARTS = {
    "ability", "able", "age", "aged", "ally", "ant", "ate", "ation", "dom",
    "ed", "er", "est", "ful", "fully", "hood", "ible", "ing", "ion", "ish",
    "ism", "ist", "ity", "less", "like", "ly", "maker", "man", "ment",
    "ness", "or", "ous", "ship", "sion", "tion", "ward", "wise",
}


def load_lines(path):
    if not path.exists():
        return []

    with path.open("r") as f:
        return [line.strip() for line in f if line.strip()]


def save_lines(path, lines):
    with path.open("w") as f:
        for line in lines:
            f.write(line + "\n")


def main():
    natural_rows = load_lines(NATURAL_PATH)
    pattern_rows = load_lines(PATTERN_PATH)

    kept_natural = []
    moved_to_pattern = []

    for line in natural_rows:
        left, right, full_word = line.split("\t")

        if right in SUFFIX_LIKE_PARTS:
            moved_to_pattern.append(line)
        else:
            kept_natural.append(line)

    combined_pattern = []
    seen = set()

    for line in pattern_rows + moved_to_pattern:
        if line in seen:
            continue

        combined_pattern.append(line)
        seen.add(line)

    #save_lines(BACKUP_PATH, natural_rows)
    save_lines(NATURAL_PATH, kept_natural)
    save_lines(PATTERN_PATH, combined_pattern)

    #print(f"Backed up original natural rows to {BACKUP_PATH}")
    print(f"Kept natural rows: {len(kept_natural)}")
    print(f"Moved to pattern_only: {len(moved_to_pattern)}")
    print(f"Pattern-only rows now: {len(combined_pattern)}")


if __name__ == "__main__":
    main()
