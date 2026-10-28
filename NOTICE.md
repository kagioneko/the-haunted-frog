# Notice

## Third-party components

### CSE-Frog (`cse-frog` on PyPI)

- What it is: the learning and generation engine (Hebbian-style next-symbol model). All training and prediction in this project is done by CSE-Frog.
- How it is used: an unmodified runtime dependency, pinned to `cse-frog==0.2.0` and installed from PyPI. Only its public API is called: `Frog(...)`, `learn`, `generate`, `probabilities`, `save`, `load`. No CSE-Frog source code is copied into this repository.
- Author: Emilia Lab / Kagino Neko (kagioneko)
- License: MIT. Copyright (c) 2026 Emilia Lab / 鍵乃ねこ (kagioneko)
- Source: https://github.com/kagioneko/cse-frog
- Package: https://pypi.org/project/cse-frog/

CSE-Frog's MIT license text ships inside the installed package. This repository is MIT-licensed too (see `LICENSE`).

## Original material in this repository

- `data/halloween.txt`, `data/control.txt` and `data/halloween_lexicon.txt` were written for this project.
- The ASCII frog in `haunted_frog/terminal.py` was drawn for this project.

## Development tools

The code and documentation were written with an AI coding assistant (Claude Code).
Every reported number comes from running the code in this repository, and the experiment can be re-run with `python -m haunted_frog evaluate`.
