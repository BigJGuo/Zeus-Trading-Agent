"""Prompt loader for agent system prompts.

Each agent role has its own Markdown file under `zeus/llm/prompts/`.
`load_prompt('day_trader')` reads it, interpolates `{shared_header}`
from `common/shared_header.md`, and returns the composed system prompt.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_PROMPT_DIR = Path(__file__).resolve().parent
_COMMON_DIR = _PROMPT_DIR / "common"


@lru_cache(maxsize=1)
def _shared_header() -> str:
    return (_COMMON_DIR / "shared_header.md").read_text(encoding="utf-8").rstrip()


@lru_cache(maxsize=1)
def _trading_principles() -> str:
    return (_COMMON_DIR / "trading_principles.md").read_text(encoding="utf-8").rstrip()


@lru_cache(maxsize=16)
def load_prompt(role: str) -> str:
    """Load and compose the system prompt for a given agent role.

    `role` is the filename stem under zeus/llm/prompts/ — e.g.
    'day_trader', 'swing_research', 'overseer'.
    """
    path = _PROMPT_DIR / f"{role}.md"
    if not path.exists():
        raise FileNotFoundError(f"no system prompt for role {role!r} at {path}")
    template = path.read_text(encoding="utf-8")
    return (
        template
        .replace("{shared_header}", _shared_header())
        .replace("{trading_principles}", _trading_principles())
    )


def available_roles() -> list[str]:
    return sorted(
        p.stem for p in _PROMPT_DIR.glob("*.md") if p.stem != "shared_header"
    )
