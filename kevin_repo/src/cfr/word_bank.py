"""Augmented word bank builder.

Addresses the professor's guidance that our total word bank should be LARGER
than NYT's ~4,918 words. Combines:

  1. NYT NYT word bank (4,918 words) -- kept for NYT-style familiarity
  2. NLTK WordNet single-token lemmas (~30k after filtering)
  3. (Optional) Google 10k most-common English words

Filters: uppercase, alphabetic, 3-15 chars, no profanity, deduplicated.

Cached at data/cache/augmented_word_bank.json.
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# A compact blocklist of obvious profanity and slurs. WordNet and common word
# lists can contain these; we filter them out to keep generated puzzles safe.
_BLOCKLIST = {
    "FUCK", "SHIT", "CUNT", "BITCH", "ASSHOLE", "BASTARD", "DAMN",
    "PISS", "COCK", "DICK", "PUSSY", "WHORE", "SLUT", "FAG",
    "NIGGER", "NIGGA", "KIKE", "SPIC", "CHINK", "GOOK", "WETBACK",
    "RETARD", "TRANNY", "DYKE", "HOMO",
}

# Only keep words matching these character constraints.
_WORD_RE = re.compile(r"^[A-Z]+$")

_MIN_LEN = 3
_MAX_LEN = 15


def _clean(word: str) -> Optional[str]:
    """Normalize and filter a single word. Returns None if invalid."""
    if not word:
        return None
    w = word.strip().upper()
    # Replace underscores (WordNet uses multi_word_lemmas) - skip those
    if "_" in w or " " in w or "-" in w:
        return None
    if not _WORD_RE.match(w):
        return None
    if len(w) < _MIN_LEN or len(w) > _MAX_LEN:
        return None
    if w in _BLOCKLIST:
        return None
    return w


def _wordnet_vocabulary(min_count: int = 1) -> list[str]:
    """Extract cleaned single-token lemmas from NLTK WordNet.

    Filters to lemmas with tagged-corpus frequency >= min_count. This removes
    archaic/obscure words (AALBORG, AARDWOLF, etc.) that WordNet contains as
    synsets but which never appear in common English usage.

    With min_count=1, yields ~16,000 lemmas (common English vocabulary).
    """
    try:
        from nltk.corpus import wordnet
    except ImportError:
        logger.warning("NLTK not available; skipping WordNet.")
        return []

    try:
        _ = wordnet.synsets("test")
    except Exception as e:
        logger.warning(f"WordNet data not available ({e}); skipping.")
        return []

    vocab = set()
    seen = set()
    for synset in wordnet.all_synsets():
        for lemma in synset.lemmas():
            name = lemma.name()
            if name in seen:
                continue
            seen.add(name)
            if lemma.count() < min_count:
                continue
            cleaned = _clean(name)
            if cleaned:
                vocab.add(cleaned)

    logger.info(
        f"WordNet: {len(vocab)} clean single-token lemmas (min_count={min_count})"
    )
    return sorted(vocab)


def _load_google_10k_if_present(project_root: Path) -> list[str]:
    """Load the Google 10k most-common words list if present; else [].

    Looks for data/external/google_10k_english.txt (optional, not required).
    """
    candidates = [
        project_root / "data" / "external" / "google_10k_english.txt",
        project_root / "data" / "external" / "google-10000-english.txt",
    ]
    for path in candidates:
        if path.exists():
            with open(path) as f:
                raw = f.read().splitlines()
            vocab = set()
            for line in raw:
                cleaned = _clean(line.split()[0] if line.strip() else "")
                if cleaned:
                    vocab.add(cleaned)
            logger.info(f"Google 10k: {len(vocab)} cleaned words from {path.name}")
            return sorted(vocab)
    return []


def build_augmented_word_bank(
    nyt_words: list[str],
    project_root: Optional[Path] = None,
    cache_path: Optional[Path] = None,
    use_wordnet: bool = True,
    use_google_10k: bool = True,
    force_rebuild: bool = False,
) -> tuple[list[str], dict]:
    """Build and cache an augmented word bank.

    Args:
        nyt_words: the 4,918 NYT words (already uppercased).
        project_root: repo root for resolving data/external/* paths.
        cache_path: where to cache the JSON output.
        use_wordnet: include WordNet lemmas.
        use_google_10k: include Google 10k words if file is present.
        force_rebuild: ignore cache.

    Returns:
        (word_bank, composition) where composition is a dict of the per-source
        counts ready for logging, e.g.
            {"total": 15342, "nyt": 4918, "wordnet_added": 10203, "google_added": 221}
    """
    if project_root is None:
        project_root = Path(__file__).resolve().parent.parent.parent
    if cache_path is None:
        cache_path = project_root / "data" / "cache" / "augmented_word_bank.json"

    # Normalize NYT words
    nyt_set = set()
    for w in nyt_words:
        c = _clean(w)
        if c:
            nyt_set.add(c)

    # Check cache
    if cache_path.exists() and not force_rebuild:
        try:
            with open(cache_path) as f:
                cached = json.load(f)
            if cached.get("nyt_count") == len(nyt_set) and cached.get("words"):
                bank = cached["words"]
                comp = cached.get("composition", {"total": len(bank)})
                logger.info(f"Loaded augmented word bank from cache: {len(bank)} words")
                return bank, comp
        except Exception as e:
            logger.warning(f"Cache read failed ({e}); rebuilding.")

    # Assemble sources
    wordnet_words = _wordnet_vocabulary() if use_wordnet else []
    google_words = _load_google_10k_if_present(project_root) if use_google_10k else []

    wn_added = set(wordnet_words) - nyt_set
    # Google words added beyond NYT+WordNet
    gg_added = set(google_words) - nyt_set - set(wordnet_words)

    full = nyt_set | wn_added | gg_added
    bank = sorted(full)

    composition = {
        "total": len(bank),
        "nyt": len(nyt_set),
        "wordnet_added": len(wn_added),
        "google_added": len(gg_added),
    }

    # Cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(
            {
                "nyt_count": len(nyt_set),
                "composition": composition,
                "words": bank,
            },
            f,
        )
    logger.info(f"Built augmented word bank: {composition}")
    return bank, composition
