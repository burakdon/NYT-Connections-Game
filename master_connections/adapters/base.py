
# ============================================================
# adapters/base.py
# ============================================================

from abc import ABC, abstractmethod
from typing import Optional
import uuid
from datetime import datetime


class PuzzleAdapter(ABC):
    """
    Abstract base for all pipeline adapters.
    Each adapter wraps one team member's pipeline and normalises
    its output into the canonical schema.
    """

    name: str = "base"

    @abstractmethod
    def generate(self) -> Optional[dict]:
        """
        Run the pipeline and return a canonical puzzle dict,
        or None if generation failed.
        """
        raise NotImplementedError

    def canonical(self,
                   yellow_words, yellow_conn,
                   green_words,  green_conn,
                   blue_words,   blue_conn,
                   purple_words, purple_conn,
                   yellow_mech='', green_mech='',
                   blue_mech='',   purple_mech='',
                   meta=None) -> dict:
        """
        Build a canonical puzzle dict from normalised parts.
        All adapters call this to produce their final output.
        """
        import random

        groups = {
            'yellow': {
                'words':      [w.upper() for w in yellow_words],
                'connection': yellow_conn,
                'mechanism':  yellow_mech,
            },
            'green': {
                'words':      [w.upper() for w in green_words],
                'connection': green_conn,
                'mechanism':  green_mech,
            },
            'blue': {
                'words':      [w.upper() for w in blue_words],
                'connection': blue_conn,
                'mechanism':  blue_mech,
            },
            'purple': {
                'words':      [w.upper() for w in purple_words],
                'connection': purple_conn,
                'mechanism':  purple_mech,
            },
        }

        all_words = (
            groups['yellow']['words'] + groups['green']['words'] +
            groups['blue']['words']   + groups['purple']['words']
        )
        board = all_words.copy()
        random.shuffle(board)

        return {
            'id':           str(uuid.uuid4()),
            'source':       self.name,
            'generated_at': datetime.utcnow().isoformat(),
            'board':        board,
            'groups':       groups,
            'meta':         meta or {},
        }