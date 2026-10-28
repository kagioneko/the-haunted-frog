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
    ended: bool            # True if the frog chose <END>; False if cut at max_len
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
        # frog.generate stops early only when it picks <END>
        ended=len(tokens) < max_len,
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
