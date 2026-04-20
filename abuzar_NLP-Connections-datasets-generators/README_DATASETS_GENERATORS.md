# NLP Connections Datasets + Generators

This folder intentionally excludes the web interface and server files.

It contains only:

- final puzzle-picking scripts,
- generated datasets,
- extractor/generator scripts,
- model/training artifacts used by the generators.

## Final Puzzle Scripts

Use `pick_one_puzzle.py` to generate one puzzle from the current databases:

```bash
python pick_one_puzzle.py
```

Use a seed for reproducible output:

```bash
python pick_one_puzzle.py --seed 42
```

Save one puzzle to JSON:

```bash
python pick_one_puzzle.py --seed 42 --output one_puzzle.json
```

Generate a larger puzzle dataset:

```bash
python generate_puzzle_pool.py --count 1000 --output data/puzzle_pool.json
```

## Final Selection Algorithm

The final picker:

1. Loads yellow, green, blue, and purple candidate-group CSVs.
2. Keeps purple rows only when `predicted_label == keep` and `keep_probability >= 0.80`.
3. Randomly picks one group from each color.
4. Rejects a puzzle if any displayed word repeats across categories.
5. Shuffles the final 16 displayed words.
6. Returns display words plus answer groups.
7. Adds a human-readable `title` while preserving the original machine-readable `answer`.

Example group output:

```json
{
  "category": "purple",
  "mechanism": "compound_suffix",
  "answer": "way",
  "title": "___ WAY",
  "words": ["free", "high", "sub", "water"]
}
```

## Current Dataset Counts

- Yellow candidate groups: 1255
- Green candidate groups: 2784
- Blue candidate groups: 853
- Raw purple candidate groups: 1266
- Scored purple candidate groups: 1266

## Folder Map

- `pick_one_puzzle.py`: final one-puzzle generator.
- `generate_puzzle_pool.py`: final multi-puzzle JSON generator.
- `data/`: generated CSVs, model artifacts, word lists, and intermediate datasets.
- `generators/`: scripts that create candidate groups or score/filter them.
- `extractors/`: scripts that extract compounds, anagrams, verb lists, and associations.
- `terminal_game/`: original terminal game file for reference.
- `references/`: original project README/PDF.

## Notes

- The final scripts look in `data/` first, then fall back to the folder root.
- Every generated puzzle has 16 unique displayed words.
- Groups may repeat across a large puzzle pool because the category databases are finite.
- The purple classifier is a quality filter trained from Claude-labeled examples, not a perfect truth label.
