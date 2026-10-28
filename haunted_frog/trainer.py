"""Training a CSE-Frog on the Halloween corpus, and saving / loading it."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

from cse import Frog

from .corpus import DEFAULT_CORPUS, UNITS, file_sha256, read_lines, to_sequences

DEFAULT_MODEL_DIR = Path("models")

# CSE-Frog's own default is top_k_edges=3, which makes the word-level frog say
# "THE GHOST" / "THE MACHINE" over and over. 16 lets more of the corpus through.
# Chosen by looking at seeds 0-39 (see docs/EXPERIMENT.md); the experiment
# itself is scored on different seeds.
DEFAULT_FROG_CONFIG = {"top_k_edges": 16}


@dataclass
class TrainInfo:
    corpus: str
    corpus_sha256: str
    lines: int
    vocabulary: int
    unit: str
    epochs: int
    frog_config: dict
    cse_frog_version: str
    trained_at: str


def cse_frog_version() -> str:
    return version("cse-frog")


def train(corpus=DEFAULT_CORPUS, unit: str = "word", epochs: int = 5, frog_config=None):
    """Teach a fresh frog every line of `corpus`. Returns (frog, TrainInfo).

    frog_config: keyword settings for cse.Frog (None = DEFAULT_FROG_CONFIG, {} = CSE-Frog defaults).
    """
    frog_config = dict(DEFAULT_FROG_CONFIG if frog_config is None else frog_config)
    if unit not in UNITS:
        raise ValueError(f"unit must be one of {UNITS}, got {unit!r}")
    if not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be an integer >= 1")
    lines = read_lines(corpus)
    sequences = to_sequences(lines, unit)
    frog = Frog(**frog_config).learn(sequences, epochs=epochs)
    info = TrainInfo(
        corpus=Path(corpus).name,
        corpus_sha256=file_sha256(corpus),
        lines=len(lines),
        vocabulary=len({tok for seq in sequences for tok in seq}),
        unit=unit,
        epochs=epochs,
        frog_config=frog_config,
        cse_frog_version=cse_frog_version(),
        trained_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    return frog, info


def model_paths(model_dir, unit: str):
    model_dir = Path(model_dir)
    return model_dir / f"frog-{unit}.cse", model_dir / f"frog-{unit}.json"


def save(frog, info: TrainInfo, model_dir=DEFAULT_MODEL_DIR):
    """Write the frog (CSE-Frog's own JSON format, no pickle) plus a metadata file."""
    model_path, meta_path = model_paths(model_dir, info.unit)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    frog.save(model_path)
    meta_path.write_text(json.dumps(asdict(info), indent=2) + "\n", encoding="utf-8")
    return model_path, meta_path


def load(model_dir=DEFAULT_MODEL_DIR, unit: str = "word"):
    model_path, meta_path = model_paths(model_dir, unit)
    if not model_path.exists():
        raise FileNotFoundError(
            f"No trained frog at {model_path}. Run: python -m haunted_frog train --unit {unit}"
        )
    info = TrainInfo(**json.loads(meta_path.read_text(encoding="utf-8")))
    return Frog.load(model_path), info
