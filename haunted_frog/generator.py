"""Letting the trained frog speak, and recording exactly what it said."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from .corpus import detokenize, tokenize

DEFAULT_MAX_LEN = {"word": 12, "char": 60}


@dataclass
class Utterance:
    text: str              # exactly what the model produced, detokenized
    tokens: List[str]
    unit: str
    seed: Optional[int]
    prompt: str
    max_len: int
    stopped_before_limit: bool  # stopped before max_len (with cse-frog 0.2.0: the frog picked <END>)
    greedy: bool = False


def generate(frog, unit: str, *, seed=None, prompt: str = "", max_len: Optional[int] = None,
             greedy: bool = False) -> Utterance:
    """One utterance. Nothing here edits, filters or "fixes" the model output."""
    if max_len is None:
        max_len = DEFAULT_MAX_LEN[unit]
    if not isinstance(max_len, int) or max_len < 1:
        raise ValueError("max_len must be an integer >= 1")
    prefix = tokenize(" ".join(prompt.upper().split()), unit) if prompt else []
    try:
        tokens = frog.generate(prefix, n=max_len, greedy=greedy, seed=seed)
    except ValueError as exc:
        # CSE-Frog refuses symbols it has never learned (its message is in Japanese).
        raise ValueError(
            f"The frog has never seen part of the prompt {prompt!r}. "
            "Use words (or letters) that appear in the corpus."
        ) from exc
    tokens = [str(t) for t in tokens]
    return Utterance(
        text=detokenize(tokens, unit),
        tokens=tokens,
        unit=unit,
        seed=seed,
        prompt=prompt,
        max_len=max_len,
        # In cse-frog 0.2.0, Frog.generate leaves its loop early only when the sampled
        # symbol is <END>. An untrained frog has no candidates, and its probabilities()
        # are then {<END>: 1.0}, so its silence is also an <END> pick (both are tested).
        stopped_before_limit=len(tokens) < max_len,
        greedy=greedy,
    )


def generate_many(frog, unit: str, seeds: Sequence[int], **kwargs) -> List[Utterance]:
    return [generate(frog, unit, seed=s, **kwargs) for s in seeds]


def append_log(path, utterances, extra: Optional[dict] = None) -> None:
    """Append utterances as JSON lines (one per utterance)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for u in utterances:
            row = asdict(u)
            if extra:
                row.update(extra)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
