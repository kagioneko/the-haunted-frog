"""The before/after experiment.

Every condition is measured with the same seeds, the same max length and the
same metrics. Samples in the report are the first seeds in order (0, 1, 2, ...),
not hand-picked ones.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from cse import Frog

from .corpus import CONTROL_CORPUS, DEFAULT_CORPUS, read_lexicon, read_lines
from .generator import DEFAULT_MAX_LEN, generate_many
from .trainer import cse_frog_version, train

# Seeds 0-39 were used while choosing top_k_edges, so scoring uses 1000-1099.
FIRST_SEED = 1000
DEFAULT_SEEDS = range(FIRST_SEED, FIRST_SEED + 100)
SAMPLES_PER_CONDITION = 10


def _ratio(num, den):
    return round(num / den, 4) if den else 0.0


def _distinct(seqs, n):
    grams = [tuple(s[i:i + n]) for s in seqs for i in range(len(s) - n + 1)]
    return _ratio(len(set(grams)), len(grams))


def metrics(utterances, corpus_lines, lexicon):
    """Score a list of Utterances. Words are always whitespace-split text,
    so word- and char-level frogs are measured on the same scale."""
    texts = [u.text.strip() for u in utterances]
    nonempty = [t for t in texts if t]
    words = [t.split() for t in nonempty]
    all_words = [w for ws in words for w in ws]
    corpus_set = set(corpus_lines)
    corpus_vocab = {w for line in corpus_lines for w in line.split()}
    adjacent = [(a, b) for ws in words for a, b in zip(ws, ws[1:])]
    return {
        "n": len(utterances),
        "empty_rate": _ratio(len(texts) - len(nonempty), len(texts)),
        "ended_rate": _ratio(sum(u.ended for u in utterances), len(utterances)),
        "mean_words": round(sum(map(len, words)) / len(texts), 3) if texts else 0.0,
        "halloween_word_rate": _ratio(sum(w in lexicon for w in all_words), len(all_words)),
        "utterances_with_halloween_word": _ratio(
            sum(any(w in lexicon for w in ws) for ws in words), len(texts)),
        "real_word_rate": _ratio(sum(w in corpus_vocab for w in all_words), len(all_words)),
        "copied_corpus_line_rate": _ratio(sum(t in corpus_set for t in nonempty), len(nonempty)),
        "novel_line_rate": _ratio(sum(t not in corpus_set for t in nonempty), len(nonempty)),
        "unique_utterance_rate": _ratio(len(set(nonempty)), len(nonempty)),
        "distinct_1": _distinct(words, 1),
        "distinct_2": _distinct(words, 2),
        "immediate_repeat_rate": _ratio(sum(a == b for a, b in adjacent), len(adjacent)),
    }


def run_condition(name, corpus, unit, epochs, seeds, lexicon, frog_config=None):
    frog, info = train(corpus, unit=unit, epochs=epochs, frog_config=frog_config)
    utts = generate_many(frog, unit, seeds, max_len=DEFAULT_MAX_LEN[unit])
    lines = read_lines(corpus)
    top = Counter(u.text for u in utts if u.text.strip()).most_common(5)
    return {
        "name": name,
        "corpus": Path(corpus).name,
        "unit": unit,
        "epochs": epochs,
        "frog_config": info.frog_config,
        "vocabulary": info.vocabulary,
        "max_len": DEFAULT_MAX_LEN[unit],
        "metrics": metrics(utts, lines, lexicon),
        "most_common": [{"text": t, "count": c} for t, c in top],
        "samples": [{"seed": u.seed, "text": u.text, "ended": u.ended}
                    for u in utts[:SAMPLES_PER_CONDITION]],
    }


def untrained_probe(seeds, lexicon):
    """'Before' in the literal sense: a brand-new Frog() with nothing learned."""
    frog = Frog()
    utts = generate_many(frog, "word", seeds, max_len=DEFAULT_MAX_LEN["word"])
    try:
        frog.generate(["GHOST"], n=5, seed=0)
        prompt_result = "accepted"
    except ValueError:
        prompt_result = "rejected: unknown symbol"
    return {
        "name": "untrained",
        "metrics": metrics(utts, [], lexicon),
        "prompt_GHOST": prompt_result,
    }


def run_experiment(seeds=DEFAULT_SEEDS, corpus=DEFAULT_CORPUS, control=CONTROL_CORPUS):
    seeds = list(seeds)
    lexicon = read_lexicon()
    conditions = []
    for unit in ("word", "char"):
        conditions.append(run_condition(f"control-{unit}", control, unit, 5, seeds, lexicon))
        conditions.append(run_condition(f"halloween-{unit}-e5-k3(cse default)", corpus, unit, 5,
                                        seeds, lexicon, frog_config={}))
        for epochs in (1, 5, 20):
            conditions.append(
                run_condition(f"halloween-{unit}-e{epochs}", corpus, unit, epochs, seeds, lexicon))
    corpus_words = [w for line in read_lines(corpus) for w in line.split()]
    return {
        "cse_frog_version": cse_frog_version(),
        # reference point: how Halloween-heavy the training text itself is
        "corpus_halloween_word_rate": _ratio(sum(w in lexicon for w in corpus_words),
                                             len(corpus_words)),
        "seeds": [seeds[0], seeds[-1]] if seeds else [],
        "n_seeds": len(seeds),
        "untrained": untrained_probe(seeds, lexicon),
        "conditions": conditions,
    }


TABLE_COLUMNS = [
    ("halloween_word_rate", "Halloween words"),
    ("real_word_rate", "Real words"),
    ("novel_line_rate", "Novel lines"),
    ("unique_utterance_rate", "Unique"),
    ("distinct_2", "Distinct-2"),
    ("immediate_repeat_rate", "Repeats"),
    ("ended_rate", "Ended"),
    ("mean_words", "Mean words"),
]


def to_markdown(result) -> str:
    lo, hi = result["seeds"]
    out = [f"cse-frog {result['cse_frog_version']}, {result['n_seeds']} seeds per condition "
           f"(seeds {lo}-{hi}). Conditions without a k suffix use top_k_edges=16.\n"]
    head = "| Condition | " + " | ".join(label for _, label in TABLE_COLUMNS) + " |"
    out += [head, "|" + "---|" * (len(TABLE_COLUMNS) + 1)]
    for c in result["conditions"]:
        m = c["metrics"]
        out.append(f"| {c['name']} | " + " | ".join(str(m[k]) for k, _ in TABLE_COLUMNS) + " |")
    out.append(f"\nHalloween-word rate of the training corpus itself: "
               f"{result['corpus_halloween_word_rate']}")
    u = result["untrained"]
    out.append(f"\nUntrained Frog(): empty_rate={u['metrics']['empty_rate']}, "
               f"prompt 'GHOST' -> {u['prompt_GHOST']}\n")
    for c in result["conditions"]:
        out.append(f"\n**{c['name']}** (first {SAMPLES_PER_CONDITION} seeds, unedited):\n")
        out += [f"- `{s['seed']}`: {s['text'] or '(silence)'}" + ("" if s["ended"] else " ✂")
                for s in c["samples"]]
    return "\n".join(out) + "\n"


def write(result, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path = out_path.with_suffix(".md")
    md_path.write_text(to_markdown(result), encoding="utf-8")
    return out_path, md_path
