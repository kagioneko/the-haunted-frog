# Architecture

```text
haunted_frog/
  corpus.py     read_lines (comments, case, spacing) · tokenize (word|char) · to_sequences · detokenize
  trainer.py    train() -> (cse.Frog, TrainInfo) · save()/load()  models/frog-<unit>.cse + .json
  generator.py  generate() -> Utterance(text, tokens, seed, max_len, stopped_before_limit, ...) · JSONL logging
  terminal.py   haunt(): Haunted Terminal (ASCII frog, typewriter, narration, colours)
  evaluator.py  metrics() · run_experiment() · markdown report
  __main__.py   CLI: train | haunt | generate | evaluate
```

## Where the AI is

All learning and prediction is done by **CSE-Frog** (`from cse import Frog`).
This project supplies the data, calls CSE-Frog's public API, and then displays or scores what comes back.

| Step | CSE-Frog call | This project |
|---|---|---|
| learn | `Frog(top_k_edges=16).learn(list_of_sequences, epochs=E)` | builds one sequence per corpus line |
| speak | `frog.generate(prefix, n=max_len, seed=s)` | converts the tokens to text and records `seed`, `max_len` and whether `<END>` was reached |
| persist | `frog.save(path)` / `Frog.load(path)` | adds a metadata JSON file (corpus SHA-256, unit, epochs, settings, cse-frog version) |
| explain | `frog.probabilities(prefix)` | used in tests (e.g. `<END>` follows `HAPPY HALLOWEEN`) |

CSE-Frog, briefly: each symbol is a node. Learning strengthens directed links between symbols that follow each other (Hebbian-style), plus a memory of the previous *pair* of symbols.
Prediction spreads activation from the recent symbols, scores the candidates, and turns the scores into probabilities (softmax with temperature).
`top_k_edges` limits both how many links spread activation and how many candidates are kept.
There is no attention, no embedding and no gradient descent.

## Design rules

- **Model output is never edited.** `generate()` returns the frog's tokens joined by a space (word) or with nothing (char). There is no filtering, no profanity list and no "best-of" re-ranking. Empty output is shown as `(silence)`.
- **Narration is visibly separate.** It is dim, has no frog marker and comes from a fixed list in `terminal.py`. It is chosen with its own RNG, which never touches the model.
- **Reproducible.** Every frog line shows its seed. CSE-Frog draws from a private `random.Random(seed)`, so the same model plus the same seed gives the same text (tested). A haunt session with `--seed S` replays exactly.
- **Offline.** No network imports. `tests/test_offline.py` blocks sockets and still trains, saves, loads and generates.
- **Safe model files.** CSE-Frog saves JSON and numeric arrays, not pickle.
