# Experiment: what does Halloween training change?

Reproduce:

```bash
python -m haunted_frog evaluate      # ~1.5 min on one CPU core
```

The output goes to `results/experiment.json` and `results/experiment.md`.
The run reported here is saved as `results/example_results.json`.

## Question

Does training CSE-Frog on a Halloween corpus make it speak Halloween?
Does it only parrot the corpus, or does it recombine it?
How do training length, unit (word or character) and one sampling setting change that?

## Why there is a control model

The spec asked for a "before vs after" comparison.
A literal *before* model (`Frog()` with nothing learned) cannot be compared on the same metrics:

- with an empty prompt it says nothing (100% empty output), and
- given any word it raises an error ("unknown symbol").

That result is recorded as the `untrained` entry.
For a fair comparison we also train a **control frog** on `data/control.txt`.
That corpus has the same size and shape as the Halloween one (60 short lines that share common words such as THE, IS, IN, I AM and SAID THE), but its vocabulary is everyday words.
A test checks that the control corpus contains none of the lexicon words.

## Setup

| | |
|---|---|
| Engine | cse-frog 0.2.0 (PyPI), pinned |
| Corpora | `data/halloween.txt` (60 lines, 93 distinct words, 25 distinct characters), `data/control.txt` (60 lines) |
| Units | `word` (whitespace-split) and `char` (one symbol per character, including spaces) |
| Epochs | 1, 5, 20 (Halloween). The control uses 5. |
| Settings | `top_k_edges=16`, all other CSE-Frog settings at their defaults (e.g. `temperature=0.8`). The "k3" rows use pure CSE-Frog defaults (`top_k_edges=3`). |
| Generation | empty prompt, sampled (not greedy). Max 12 tokens (word) or 60 (char). Stops at `<END>`. |
| Seeds | 1000-1099: the same 100 seeds for every condition |
| Lexicon | `data/halloween_lexicon.txt` (30 words), written before the experiment and not tuned afterwards |

### How `top_k_edges=16` was chosen (disclosed)

CSE-Frog's default `top_k_edges=3` keeps only the top-3 candidates at each step.
On this corpus that produced almost nothing but `THE GHOST` / `THE MACHINE`.
We tried k ∈ {3, 8, 16} × temperature ∈ {0.8, 1.2} on **seeds 0-39** and looked at the outputs. k=16 with the default temperature gave the most varied word-level output.
To avoid scoring on the samples used for that choice, every number below comes from **seeds 1000-1099**.
The k3 rows are kept in the table so the effect of this choice is visible.

## Metrics

All metrics are computed on whitespace-split words of the output text, so word- and char-level frogs are measured on the same scale.

| Metric | Meaning |
|---|---|
| Halloween words | share of generated words that are in the lexicon |
| Real words | share of generated words that appear anywhere in the training corpus |
| Novel lines | share of non-empty outputs that are **not** an exact corpus line. This does not imply grammaticality. |
| Unique | distinct outputs / non-empty outputs |
| Distinct-2 | distinct word bigrams / all word bigrams |
| Repeats | share of adjacent word pairs that are the same word (`IN IN`) |
| Ended | share of outputs that stopped by choosing `<END>` rather than hitting max length |

## Results (seeds 1000-1099, 100 per condition)

| Condition | Halloween words | Real words | Novel lines | Unique | Distinct-2 | Repeats | Ended | Mean words |
|---|---|---|---|---|---|---|---|---|
| control-word | 0.0 | 1.0 | 0.95 | 0.59 | 0.2471 | 0.0 | 0.99 | 3.63 |
| halloween-word-e5-k3 (cse default) | 0.1502 | 1.0 | 0.8 | 0.2 | 0.0704 | 0.0 | 1.0 | 3.13 |
| halloween-word-e1 | 0.1847 | 1.0 | 0.83 | 0.81 | 0.3312 | 0.0 | 0.99 | 4.17 |
| halloween-word-e5 | 0.1549 | 1.0 | 0.91 | 0.57 | 0.2351 | 0.0 | 0.99 | 3.68 |
| halloween-word-e20 | 0.194 | 1.0 | 0.94 | 0.38 | 0.2312 | 0.0 | 1.0 | 2.99 |
| control-char | 0.0 | 0.5872 | 1.0 | 0.53 | 0.3084 | 0.0969 | 1.0 | 3.27 |
| halloween-char-e5-k3 (cse default) | 0.0 | 0.778 | 1.0 | 0.83 | 0.102 | 0.285 | 0.84 | 8.65 |
| halloween-char-e1 | 0.0017 | 0.3645 | 1.0 | 0.99 | 0.8474 | 0.0522 | 0.84 | 5.98 |
| halloween-char-e5 | 0.0 | 0.7293 | 1.0 | 0.92 | 0.2684 | 0.2518 | 0.74 | 9.42 |
| halloween-char-e20 | 0.0 | 0.7154 | 1.0 | 0.91 | 0.2649 | 0.2928 | 0.79 | 8.89 |

- Halloween-word rate of the training corpus itself: **0.2255**.
- Untrained `Frog()`: 100% empty output. The prompt `GHOST` is rejected as an unknown symbol.

### Unedited samples: the first 10 seeds of a condition, not picked by hand

**halloween-word-e5** (the default `haunt` model):
`THE DARK` · `THE FROG IS FLICKERING` · `THE GHOST` · `THE GHOST IS LAUGHING` · `THE NIGHT IS HUNGRY` · `THE OWL SAYS WHO` · `THE GHOST` · `THE ZOMBIE IS FLICKERING` · `THE FOG IS HUNGRY` · `THE MACHINE`

**halloween-word-e1**:
`COLD WIND AT MIDNIGHT` · `COLD WIND AT MIDNIGHT` · `A SPIDER IN THE WINDOW` · `I AM HUNGRY` · `THE NIGHT IS QUIET` · `I WANT A WITCH ON A GHOST IN THE WIRES` · `THE MACHINE` · `SOMETHING IS HUNGRY` · `A FROG` · `THE WITCH IS AWAKE`

**control-word**:
`THE BAKER IS SINGING` · `THE DOG` · `THE WINDOW` · `THE BIRD ATE THE KITCHEN` · `THE DAY IS READING` · `THE FLOWER IS SLEEPING` · `THE DOG IS LONG` · `THE LIBRARY IS SLEEPING` · `THE RAIN IS READING` · `THE KITCHEN`

**halloween-char-e5**: the 10 seeds are mostly variations of
`THE IN IN IN ING`, `THE IS IN IN ING ING`, `THE INGHE INGHE IN IN IN WIS INDYS ING`.
Lines marked ✂ in `results/experiment.md` hit the 60-character limit.

**halloween-char-e1**: `HE S L THE AUNIGIC`, `ADOVEBOMAMIDR THE IN INTHE SPKI W`, `N`, `C`, ...

## Findings

1. **Training is what makes it speak at all, and the corpus decides what it says.** Untrained: silence. Control corpus: 0% Halloween words. Halloween corpus: 15-19%.
2. **It recombines rather than recites.** 83-94% of word-level outputs are not exact corpus lines. The frog swaps the subject and predicate across lines that share `THE ... IS ...` (`THE ZOMBIE IS FLICKERING`, `THE GHOST IS IN THE FROG`). Some recombinations are not grammatical.
3. **It amplifies the most common word.** `THE` is the corpus's most frequent word (41 of 235 words) and starts 23 of 60 lines (38%), yet it starts almost every word-level output. The Halloween-word rate stays *below* the corpus's own 22.6%, because function words (THE, IS) are over-produced.
4. **More epochs mean less variety.** Unique outputs fall from 0.81 (e1) to 0.57 (e5) to 0.38 (e20), and the outputs get shorter (4.17 to 2.99 words). The strongest paths get stronger and the frog falls into a few attractors (`THE GHOST`, `THE MACHINE`).
5. **The candidate cutoff matters more than epochs.** At the CSE-Frog default `top_k_edges=3`, only 20% of outputs are unique (k=16: 57%) at the same epoch count.
6. **The character-level frog cannot hold a word together.** It almost never produces a Halloween word (0-0.2%). It falls into an `IN IN ING` loop: `THE`, `IS` and `IN` are among the most frequent character runs in the corpus, and many lines end in `-ING`. About two characters of context is not enough to escape them. More training raises "real word" share (0.36 to 0.73) mainly by making it repeat `THE`/`IN`/`IS`. This is the clearest demonstration in the project of why real language models need long context.

## Threats to validity

- The corpus and lexicon were written by the same author, and the control corpus was written to mirror the Halloween corpus. The comparison is between two hand-made corpora, not natural text.
- 100 samples per condition, with no confidence intervals. Treat small differences (e.g. Halloween words 0.15 vs 0.19) as noise.
- `top_k_edges` was chosen after looking at outputs (on different seeds). Other settings were left at their defaults and not explored.
- Metrics count surface words only. Nothing here measures meaning or grammaticality.
