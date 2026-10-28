"""Loading the corpus and turning lines into CSE-Frog sequences."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import List

UNITS = ("word", "char")

PACKAGE_DIR = Path(__file__).resolve().parent
DATA_DIR = PACKAGE_DIR.parent / "data"
DEFAULT_CORPUS = DATA_DIR / "halloween.txt"
CONTROL_CORPUS = DATA_DIR / "control.txt"
LEXICON = DATA_DIR / "halloween_lexicon.txt"


def read_lines(path) -> List[str]:
    """Read one phrase per line. Blank lines and '#' comments are skipped.

    Lines are upper-cased and inner whitespace is collapsed to single spaces,
    so "trick  or treat" and "TRICK OR TREAT" are the same training line.
    """
    lines = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(" ".join(line.upper().split()))
    if not lines:
        raise ValueError(f"No training lines found in {path}")
    return lines


def tokenize(line: str, unit: str) -> List[str]:
    """Split one line into the symbols the frog learns.

    word: "THE GHOST" -> ["THE", "GHOST"]
    char: "THE GHOST" -> ["T", "H", "E", " ", "G", ...]
    """
    if unit == "word":
        return line.split()
    if unit == "char":
        return list(line)
    raise ValueError(f"unit must be one of {UNITS}, got {unit!r}")


def to_sequences(lines: List[str], unit: str) -> List[List[str]]:
    """One list per line.

    This shape matters: CSE-Frog reads a *flat* list of strings as ONE sequence
    (one element = one symbol), which would glue every line together into a
    single run-on sequence. A list of lists keeps each line separate, so the
    frog learns where each phrase ends.
    """
    return [tokenize(line, unit) for line in lines]


def detokenize(tokens, unit: str) -> str:
    return " ".join(tokens) if unit == "word" else "".join(tokens)


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_lexicon(path=LEXICON) -> set:
    return set(read_lines(path))
