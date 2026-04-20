# Connection Forge Handoff

This is a playable Connections-style puzzle MVP with a multi-agent Claude group generator, a local verified group bank, a puzzle assembler, and a NYT archive hash guard.

## Current State

- Playable puzzles: `data/puzzles.json` currently has 25 puzzles.
- Verified groups: `data/groups.json` currently has 64 reusable groups.
- Concept inspiration: `data/concept_inspiration.json` currently has 648 inspiration concepts.
- NYT guard: `data/nyt_blocklist.json` stores hashes used to avoid past NYT Connections puzzles.

## Setup

1. Open Terminal and go to the project folder:

   ```bash
   cd "/path/to/repo"
   ```

2. Create a local environment file:

   ```bash
   cp .env.example .env.local
   ```

3. Put a Claude API key in `.env.local`:

   ```text
   ANTHROPIC_API_KEY=your_claude_api_key_here
   ```

4. Optional dependency install:

   ```bash
   python3 -m pip install -r requirements.txt
   ```

   The web server and most MVP scripts use the Python standard library. `certifi` helps Claude HTTPS calls on some macOS Python installs.

## Run The Web App

```bash
python3 web_server.py --port 8000
```

Then open:

```text
http://127.0.0.1:8000
```

## Main Workflow

Generate reusable groups with Claude:

```bash
python3 generate_groups.py --count 3 --difficulty mixed --max-attempts 1 --max-empty-batches 1
```

Build playable puzzles from the verified group bank:

```bash
python3 generate_puzzles.py --count 5 --strategy group-bank --difficulty easy --batch-size 1
```

The first command uses the Claude API. The second command does not call Claude.

## Web Workflow

In the Generate tab:

1. Click `Generate Groups` to ask Claude for reusable category groups.
2. Click `Build Puzzles` to assemble playable boards from approved groups.
3. Go back to Play and click `New Puzzle`.

## Group Generation Rules

- Generate one category group at a time, not full puzzles.
- Each group must have exactly four playable words.
- All four words must satisfy one clean rule.
- Category labels should be polished reveal titles, not clue sentences.
- Each group needs `difficulty`, `mechanism_family`, and `concept_key`.
- `concept_key` should expose the real concept so renamed repeats can be caught.
- Do not repeat existing category concepts or exact four-word groups.
- Reject groups that overlap three or more words with an existing group.
- Concept inspiration is a study guide, not a menu. Do not copy exact titles, concept keys, or near-renames.
- Avoid fake wordplay, scratch explanations, and invalid homophone/hidden-letter claims.

## Important Files

- `web_server.py`: local web server and API.
- `web/`: frontend app.
- `agents/group_agents.py`: Claude Group Generator and Group Auditor.
- `agents/group_bank.py`: verified group storage and puzzle assembly.
- `agents/puzzle_validator.py`: deterministic local validation.
- `agents/nyt_guard.py`: hash checks against archived NYT puzzles.
- `data/groups.json`: verified category groups.
- `data/puzzles.json`: playable puzzle bank.
- `data/concept_inspiration.json`: inspiration list sampled in prompts.
- `data/nyt_blocklist.json`: hashed archive guard.
- `data/latest_agent_run.json`: latest agent trace shown in Agent Lab.

## Security Note

Do not commit or share `.env.local`. It contains the local Claude API key.
