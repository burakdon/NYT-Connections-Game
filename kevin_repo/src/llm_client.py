"""LLM client abstraction with mock support for DRY_RUN mode."""

import json
import random
from typing import Optional

from src.config import (
    DRY_RUN, LLM_PROVIDER,
    OPENAI_API_KEY, OPENAI_MODEL,
    ANTHROPIC_API_KEY, CLAUDE_MODEL, CLAUDE_MAX_TOKENS,
)


class LLMClient:
    """Unified LLM client supporting OpenAI, Anthropic, and Mock."""

    def __init__(self, dry_run: Optional[bool] = None, provider: Optional[str] = None):
        self.dry_run = dry_run if dry_run is not None else DRY_RUN
        self.provider = provider or LLM_PROVIDER
        self._client = None

        if self.dry_run:
            self.provider = "mock"
            return

        if self.provider == "openai":
            from openai import OpenAI
            self._client = OpenAI(api_key=OPENAI_API_KEY)
            self.model = OPENAI_MODEL
        elif self.provider == "anthropic":
            import anthropic
            self._client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            self.model = CLAUDE_MODEL
        else:
            self.provider = "mock"

    def complete(self, system: str, user: str, temperature: float = 0.7) -> str:
        if self.provider == "mock":
            return MockLLMClient.respond(system, user)

        if self.provider == "openai":
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                response_format={"type": "json_object"},
            )
            return _strip_markdown(response.choices[0].message.content)

        if self.provider == "anthropic":
            response = self._client.messages.create(
                model=self.model,
                max_tokens=CLAUDE_MAX_TOKENS,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return _strip_markdown(response.content[0].text)

        return MockLLMClient.respond(system, user)


def _strip_markdown(text: str) -> str:
    """Strip markdown code fences (```json ... ```) that some LLMs wrap around JSON."""
    if not text:
        return text
    text = text.strip()
    if text.startswith("```"):
        # Remove opening fence (```json or just ```)
        lines = text.split("\n")
        lines = lines[1:]
        # Remove closing fence
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


class MockLLMClient:
    """Returns realistic mock responses for testing without API keys.

    Detects the type of request from the prompt content and returns
    appropriately structured JSON responses that exercise all code paths.
    """

    # Pre-built category pools: each has 8 candidate words (pipeline picks best 4 via MPNET)
    CATEGORY_POOLS = [
        {
            "category": "WAYS TO SAY HELLO",
            "words": ["HI", "HEY", "HOWDY", "GREETINGS", "SUP", "YO", "HOLA", "SALUTATIONS"],
        },
        {
            "category": "THINGS WITH WHEELS",
            "words": ["CAR", "BICYCLE", "SKATEBOARD", "WAGON", "TRUCK", "BUS", "CART", "SCOOTER"],
        },
        {
            "category": "FIRE ___",
            "words": ["TRUCK", "WORK", "FLY", "PLACE", "FIGHTER", "CRACKER", "HOUSE", "SIDE"],
        },
        {
            "category": "SHADES OF BLUE",
            "words": ["NAVY", "COBALT", "AZURE", "TEAL", "CERULEAN", "INDIGO", "SAPPHIRE", "SKY"],
        },
        {
            "category": "BREAKFAST ITEMS",
            "words": ["WAFFLE", "PANCAKE", "BACON", "TOAST", "EGGS", "CEREAL", "MUFFIN", "BAGEL"],
        },
        {
            "category": "MUSICAL INSTRUMENTS",
            "words": ["PIANO", "GUITAR", "DRUMS", "VIOLIN", "TRUMPET", "FLUTE", "CELLO", "HARP"],
        },
        {
            "category": "TAYLOR SWIFT ALBUMS",
            "words": ["FOLKLORE", "EVERMORE", "REPUTATION", "MIDNIGHTS", "FEARLESS", "RED", "LOVER", "DEBUT"],
        },
        {
            "category": "TYPES OF DANCE",
            "words": ["SALSA", "WALTZ", "TANGO", "BALLET", "SWING", "FOXTROT", "JAZZ", "HIP-HOP"],
        },
        {
            "category": "PLANETS",
            "words": ["MARS", "VENUS", "JUPITER", "SATURN", "MERCURY", "NEPTUNE", "URANUS", "EARTH"],
        },
        {
            "category": "CARD GAMES",
            "words": ["POKER", "BRIDGE", "RUMMY", "SOLITAIRE", "HEARTS", "SPADES", "BLACKJACK", "WAR"],
        },
        {
            "category": "___ BALL",
            "words": ["BASE", "BASKET", "FOOT", "SNOW", "FIRE", "SOFT", "BEACH", "DODGE"],
        },
        {
            "category": "WORDS MEANING FAST",
            "words": ["QUICK", "SWIFT", "RAPID", "SPEEDY", "BRISK", "HASTY", "FLEET", "NIMBLE"],
        },
    ]

    # False group roots for the false-group pipeline
    FALSE_GROUP_ROOTS = [
        {
            "category": "THINGS THAT ARE ROUND",
            "words": ["BALL", "GLOBE", "RING", "WHEEL"],
            "alternate_meanings": {
                "BALL": "a formal dance event",
                "GLOBE": "a theater or news organization",
                "RING": "the sound a phone makes",
                "WHEEL": "to turn or rotate",
            },
        },
        {
            "category": "TYPES OF KEYS",
            "words": ["PIANO", "SKELETON", "MASTER", "FLORIDA"],
            "alternate_meanings": {
                "PIANO": "a musical instrument",
                "SKELETON": "a bare-bones version of something",
                "MASTER": "an expert or authority",
                "FLORIDA": "a chain of islands",
            },
        },
    ]

    _call_count = 0

    @classmethod
    def reset(cls):
        cls._call_count = 0

    @classmethod
    def respond(cls, system: str, user: str) -> str:
        cls._call_count += 1
        prompt_lower = (system + " " + user).lower()

        # Order matters: most specific patterns first
        if "solve this connections" in prompt_lower or "solving an nyt" in prompt_lower:
            return cls._solver_response(prompt_lower)
        if "false group" in prompt_lower or "root group" in prompt_lower:
            return cls._false_group_response(prompt_lower)
        if "given a word and its alternate meaning" in prompt_lower:
            return cls._alternate_meaning_response(prompt_lower)
        if "puzzle editor" in prompt_lower or "reviewing a connections" in prompt_lower:
            return cls._editor_response(prompt_lower)
        if "category" in prompt_lower and ("group" in prompt_lower or "word" in prompt_lower):
            return cls._group_creation_response(prompt_lower)
        if "difficulty" in prompt_lower or "rank" in prompt_lower:
            return cls._difficulty_response()

        return cls._group_creation_response(prompt_lower)

    @classmethod
    def _group_creation_response(cls, prompt_lower: str) -> str:
        idx = cls._call_count % len(cls.CATEGORY_POOLS)
        pool = cls.CATEGORY_POOLS[idx]
        return json.dumps({
            "category": pool["category"],
            "words": pool["words"],
        })

    @classmethod
    def _false_group_response(cls, prompt_lower: str) -> str:
        root = cls.FALSE_GROUP_ROOTS[cls._call_count % len(cls.FALSE_GROUP_ROOTS)]
        return json.dumps({
            "category": root["category"],
            "words": root["words"],
            "alternate_meanings": root["alternate_meanings"],
        })

    @classmethod
    def _alternate_meaning_response(cls, prompt_lower: str) -> str:
        idx = cls._call_count % len(cls.CATEGORY_POOLS)
        pool = cls.CATEGORY_POOLS[idx]
        return json.dumps({
            "category": pool["category"],
            "words": pool["words"],
        })

    @classmethod
    def _editor_response(cls, prompt_lower: str) -> str:
        return json.dumps({
            "valid": True,
            "changes": [],
            "notes": "All categories accurately describe their word groups.",
        })

    @classmethod
    def _solver_response(cls, prompt_lower: str) -> str:
        return json.dumps({
            "groups": [
                {"category": "GROUP A", "words": ["WORD1", "WORD2", "WORD3", "WORD4"]},
                {"category": "GROUP B", "words": ["WORD5", "WORD6", "WORD7", "WORD8"]},
                {"category": "GROUP C", "words": ["WORD9", "WORD10", "WORD11", "WORD12"]},
                {"category": "GROUP D", "words": ["WORD13", "WORD14", "WORD15", "WORD16"]},
            ],
            "reasoning": "Mock solver grouped words by semantic similarity.",
        })

    @classmethod
    def _difficulty_response(cls) -> str:
        return json.dumps({
            "ranking": [
                {"category": "GROUP A", "difficulty": 1},
                {"category": "GROUP B", "difficulty": 2},
                {"category": "GROUP C", "difficulty": 3},
                {"category": "GROUP D", "difficulty": 4},
            ]
        })
