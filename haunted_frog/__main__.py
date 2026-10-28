"""python -m haunted_frog {train,haunt,generate,evaluate}"""
from __future__ import annotations

import argparse
import math
import sys

from . import evaluator, trainer
from .corpus import DEFAULT_CORPUS, UNITS
from .generator import append_log, generate_many
from .terminal import haunt


def _int_at_least(lo):
    def parse(text):
        value = int(text)
        if value < lo:
            raise argparse.ArgumentTypeError(f"must be an integer >= {lo}")
        return value
    return parse


def _seconds(text):
    value = float(text)
    if not math.isfinite(value) or value < 0:
        raise argparse.ArgumentTypeError("must be a finite number >= 0")
    return value


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="haunted_frog", description="Teach a tiny frog Halloween.")
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="train a frog on the corpus and save it")
    t.add_argument("--corpus", default=str(DEFAULT_CORPUS))
    t.add_argument("--unit", choices=UNITS, default="word")
    t.add_argument("--epochs", type=_int_at_least(1), default=5)
    t.add_argument("--model-dir", default=str(trainer.DEFAULT_MODEL_DIR))
    t.add_argument("--top-k-edges", type=_int_at_least(1), default=trainer.DEFAULT_FROG_CONFIG["top_k_edges"])
    t.add_argument("--temperature", type=float, help="CSE-Frog temperature (default: CSE-Frog's)")

    h = sub.add_parser("haunt", help="open the Haunted Terminal")
    h.add_argument("--unit", choices=UNITS, default="word")
    h.add_argument("--model-dir", default=str(trainer.DEFAULT_MODEL_DIR))
    h.add_argument("--count", type=_int_at_least(0), default=8, help="lines to speak (0 = forever)")
    h.add_argument("--interval", type=_seconds, default=2.5)
    h.add_argument("--seed", type=int)
    h.add_argument("--prompt", default="", help="words the frog must continue from")
    h.add_argument("--fast", action="store_true", help="no typewriter / no pauses")
    h.add_argument("--no-halloween", action="store_true", help="plain output, no effects")
    h.add_argument("--log", help="append every line to this JSONL file")

    g = sub.add_parser("generate", help="print raw generations, one per line")
    g.add_argument("--unit", choices=UNITS, default="word")
    g.add_argument("--model-dir", default=str(trainer.DEFAULT_MODEL_DIR))
    g.add_argument("-n", type=_int_at_least(1), default=10)
    g.add_argument("--seed", type=int, default=0, help="first seed; line i uses seed+i")
    g.add_argument("--max-len", type=_int_at_least(1))
    g.add_argument("--prompt", default="")
    g.add_argument("--log")

    e = sub.add_parser("evaluate", help="run the before/after experiment")
    e.add_argument("--seeds", type=_int_at_least(1), default=100)
    e.add_argument("--first-seed", type=int, default=evaluator.FIRST_SEED)
    e.add_argument("--out", default="results/experiment.json")

    a = p.parse_args(argv)
    try:
        if a.cmd == "train":
            config = {"top_k_edges": a.top_k_edges}
            if a.temperature is not None:
                config["temperature"] = a.temperature
            frog, info = trainer.train(a.corpus, unit=a.unit, epochs=a.epochs, frog_config=config)
            model_path, meta_path = trainer.save(frog, info, a.model_dir)
            print(f"Trained on {info.lines} lines ({info.vocabulary} {info.unit} symbols), "
                  f"{info.epochs} epochs, cse-frog {info.cse_frog_version}")
            print(f"Saved {model_path} and {meta_path}")
        elif a.cmd == "haunt":
            frog, info = trainer.load(a.model_dir, a.unit)
            haunt(frog, info, count=a.count, interval=0 if a.fast else a.interval,
                  typing_delay=0 if a.fast else 0.05, seed=a.seed, prompt=a.prompt,
                  halloween=not a.no_halloween, log_path=a.log)
        elif a.cmd == "generate":
            frog, info = trainer.load(a.model_dir, a.unit)
            utts = generate_many(frog, a.unit, range(a.seed, a.seed + a.n),
                                 prompt=a.prompt, max_len=a.max_len)
            for u in utts:
                print(f"{u.seed}\t{u.text}")
            if a.log:
                append_log(a.log, utts, {"epochs": info.epochs})
        elif a.cmd == "evaluate":
            result = evaluator.run_experiment(seeds=range(a.first_seed, a.first_seed + a.seeds))
            json_path, md_path = evaluator.write(result, a.out)
            print(evaluator.to_markdown(result).split("\n**")[0])
            print(f"Wrote {json_path} and {md_path}")
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
