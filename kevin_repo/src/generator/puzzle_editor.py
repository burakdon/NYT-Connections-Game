"""Puzzle editor: reviews and fixes category names via a second LLM pass."""

import json
import logging

from src.llm_client import LLMClient
from src.generator.prompts import format_editor

logger = logging.getLogger(__name__)


class PuzzleEditor:
    """Reviews a complete puzzle and fixes inaccurate category names."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def review(self, groups: list[dict]) -> list[dict]:
        """Review and potentially fix category names.

        Returns the (possibly modified) groups list.
        """
        system, user = format_editor(groups)
        response = self.llm.complete(system, user)

        try:
            result = json.loads(response)
        except json.JSONDecodeError:
            logger.warning("Editor returned non-JSON response; keeping original groups.")
            return groups

        if not result.get("valid", True) and result.get("changes"):
            change_map = {
                c["old_category"].upper(): c["new_category"].upper()
                for c in result["changes"]
            }
            for group in groups:
                cat = group["category"].upper()
                if cat in change_map:
                    old = group["category"]
                    group["category"] = change_map[cat]
                    logger.info(f"Editor renamed '{old}' -> '{group['category']}'")

        notes = result.get("notes", "")
        if notes:
            logger.info(f"Editor notes: {notes}")

        return groups
