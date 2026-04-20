# Connection Forge Web MVP

Connection Forge is a browser-playable, Connections-style puzzle game with a Claude-backed multi-agent puzzle factory.

## Run the app

```bash
python3 web_server.py
```

Open:

```text
http://127.0.0.1:8000
```

The game works immediately from `data/puzzles.json`.

## Enable Claude generation

Create `.env.local`:

```bash
cp .env.example .env.local
```

Add your Claude API key:

```text
ANTHROPIC_API_KEY=your_key_here
```

Then run the app and use the `Generate` tab, or generate a larger bank from the terminal.

## Generate puzzles from the terminal

Generated puzzles are blocked until the NYT archive guard has hashes loaded. This is intentional: it prevents accepting a puzzle before the past-puzzle check is active.

Build the hash-only blocklist from an instructor-approved archive export:

```bash
python3 build_nyt_blocklist.py --input path/to/nyt-connections-archive.json
```

The builder also accepts copied archive text or HTML containing lines like `Yellow Group: CATEGORY: WORD, WORD, WORD, WORD`:

```bash
python3 build_nyt_blocklist.py --input path/to/archive.txt
```

The output file is `data/nyt_blocklist.json`. It stores hashes only, not raw NYT answers.

```bash
python3 build_group_bank.py
python3 generate_puzzles.py --count 100 --strategy group-bank --batch-size 1 --difficulty easy
```

The `group-bank` strategy does not call Claude. It assembles new boards from
`data/groups.json`, then runs local validation and the NYT hash guard before
saving. This is the recommended MVP path for building a playable bank quickly.

To grow the bank with LLM generation, generate verified groups first:

```bash
python3 generate_groups.py --count 5 --difficulty mixed --max-attempts 1 --max-empty-batches 1
```

This runs a Group Generator agent and a Group Auditor agent, then saves only
locally valid approved groups to `data/groups.json`. After that, assemble
playable puzzles from the larger verified group bank.

To create one fully fresh puzzle, run the group generator once per color lane
and then assemble those four fresh groups:

```bash
python3 generate_puzzles.py --count 1 --strategy fresh-puzzle --difficulty easy --batch-size 1 --max-attempts 1
```

This calls the group pipeline four times: `easy` for yellow, `medium` for
green, `hard` for blue, and `tricky` for purple. The final puzzle still runs
local puzzle validation and the NYT hash guard before saving. The fresh groups
are saved to `data/groups.json` only if the puzzle itself is accepted.

The group generator also samples a small subset from
`data/concept_inspiration.json` on each run. Those concepts are inspiration
only, not a menu. Claude should study why a sampled concept works, then create
a different sibling idea, contrast idea, narrower/broader variant, or
mechanism-shifted group. The local validator rejects exact or near-copied
concept titles and concept keys.

Group-generation rules:

- Each generated item is one category group, not a full puzzle.
- Every group must have exactly four playable answer words.
- All four words must satisfy the same clean rule.
- Category labels should be polished reveal titles, not clue sentences.
- Every group needs `difficulty`, `mechanism_family`, and `concept_key`.
- `concept_key` should identify the real concept, so renamed repeats can be caught.
- Do not repeat an existing category concept or exact four-word group.
- Avoid groups that overlap three or more words with an existing group.
- Use concept-inspiration seeds only as springboards; do not copy exact titles, concept keys, or near-renames.
- Avoid stale broad sets unless they are useful easy/yellow groups.
- Avoid anagrams unless every displayed word is common and the mapping is exact.
- For homophones, display the soundalike answer, not the target item.
- For hidden-letter groups, every explanation claim must be literally true.
- Explanations must be final answer logic only, with no scratch notes or corrections.

Recommended low-cost options:

```bash
python3 generate_puzzles.py --count 1 --strategy group-bank --difficulty easy
python3 generate_puzzles.py --count 1 --difficulty easy --strategy standard --batch-size 1 --max-attempts 2 --max-empty-batches 2 --max-review-rounds 1
python3 generate_puzzles.py --count 5 --difficulty easy --strategy standard --batch-size 1 --max-attempts 8 --max-empty-batches 3 --max-review-rounds 1
```

Recommended higher-quality option:

```bash
python3 generate_puzzles.py --count 1 --difficulty easy --strategy standard --batch-size 1 --max-attempts 2 --max-empty-batches 2 --max-review-rounds 1 --generator-model claude-opus-4-7 --reviewer-model claude-sonnet-4-6
```

You can also set `CLAUDE_GENERATOR_MODEL=claude-opus-4-7` and `CLAUDE_REVIEWER_MODEL=claude-sonnet-4-6` in `.env.local` so the shorter commands use Opus for generation and Sonnet for solver/critic review.

The CLI now defaults to the no-API group-bank assembler. The Claude strategies still stop after 6 batches total or 2 consecutive empty batches, and each batch uses 2 review rounds unless you override those settings.

The medium Tree-of-Thought strategy is experimental and more expensive. It is useful for demos or research notes, but it is not the recommended production path for building the MVP bank:

```bash
python3 generate_puzzles.py --count 1 --difficulty easy --strategy tot --max-attempts 1 --max-empty-batches 1 --max-review-rounds 2
```

## Generator modes

- `easy`: clean categories, no intentional decoy.
- `hard`: clean categories plus exactly one fair decoy cluster.

Generated groups should be ordered for the reveal colors:

```text
yellow -> green -> blue -> purple
easiest -> slightly harder -> hard -> hardest
```

Category labels should read like polished answer reveals:

```text
Prefer: Things with scales
Avoid: Can have scales
```

## Bank memory

Before Claude generates a batch, the system scans `data/puzzles.json` and sends a compact avoid-list to the Category Scout, Wordsmith, Misdirection Agent, and Editor.

The memory prevents repetition in two ways:

- Exact repeated four-word answer groups are rejected.
- Specific category labels, such as `Planets` or `Flowers`, are rejected if repeated.
- Overused generated families, such as `Shades of blue` and `Shades of green`, are treated as the same family and blocked after one appears.
- Generated groups include hidden `mechanism_family` and `concept_key` metadata so renamed repeats can be caught locally.

Reusable mechanisms can repeat with a new target:

```text
Allowed: Words before board, Words before paper
Rejected: Words before board, Words before board
```

For purple categories, simple compound/phrase mechanisms are allowed but capped for variety. The bank should not make `___ X`, `X ___`, or `Words before/after X` the default hardest category.

## Mechanism library

The generator uses `data/mechanism_library.json` as a local inspiration cache. Each run sends only a compact sampled subset of the library, not the whole list.

The library is non-exhaustive: Claude may invent new categories, but it should follow fair mechanisms like shared hidden properties, sound patterns, spelling patterns, second identities, transformations, phrase completion, semantic sets, or light trivia.

The generator also samples a few neutral words from `data/inspiration_words.json` for each run. These are loose creativity seeds only; Claude is told not to force those words into the puzzle.

Shared-property categories are prompted to use the same natural phrase test for all four words, so one answer cannot sneak in through awkward or merely technical wording.

The generator is also told to avoid using more than one same-frame `Things you can X` or `Things that can be X` shared-property group in a single board.

## Agent pipeline

The product uses separate agents coordinated by Python:

- Category Scout proposes category ideas.
- Wordsmith builds candidate puzzles.
- Misdirection Agent removes accidental traps for easy puzzles and adds one fair decoy for hard puzzles.
- Solver Agent tries to solve from shuffled words only.
- Critic Agent scores quality and fairness.
- Editor Agent repairs and finalizes.
- Local Validator rejects objective format problems.
- Local Validator also rejects generated puzzles that are too bland: more than one compound-word group, repeated same-frame can-verb groups, too many broad everyday-list categories, color-shade purple groups, plain synonym groups as purple, or too many purple compound/phrase groups across the generated bank.
- Local Validator rejects paper-style constraint problems for new generated puzzles: missing metadata, broad category labels, category label words appearing as answers, and thematically overlapping category labels.
- Local Validator checks objective wordplay leakage. For phrase and reversal groups, it rejects boards where an off-category word also satisfies the same rule, such as `STEP -> OVERSTEP`, `FLOW -> WOLF`, or `WON -> NOW`.
- Local Validator reads explanations for claim checks. It verifies hidden-word claims and common homophone direction, so a category like `Homophones of animals` rejects `GNU = new`; the displayed tile should be `NEW = gnu`.
- NYT Guard rejects generated puzzles whose board or answer grouping matches the hash blocklist.

## Experimental Tree-of-Thought Strategy

The optional `--strategy tot` path uses a medium-depth tree search before the normal review loop. It is currently set aside for cost control; the standard multi-agent pipeline is the recommended MVP generation path.

1. Category Thought Agent generates many independent four-word category thoughts.
2. ToT Local Pruner rejects repeated, malformed, or bank-conflicting thoughts before more Claude calls are spent on them.
3. ToT Thought Ranker scores the surviving thoughts by difficulty lane, mechanism fit, and stated risk.
4. Board Builder Agent assembles several candidate boards from the ranked pool.
5. ToT Board Pruner rejects complete boards with local validation errors, repeats, NYT hash matches, or objective wordplay leakage before polish.
6. The existing Misdirection, Solver, Critic, Editor, Local Validator, and NYT Guard pipeline reviews the candidates.

The latest run is saved to `data/latest_agent_run.json` and shown in the `Agent Lab` tab.
