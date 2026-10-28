"""Why does more training make the frog duller? Print the next-word
probabilities CSE-Frog assigns at 1 vs 20 epochs, plus a score breakdown.

    python scripts/diagnose.py
"""
from haunted_frog import trainer
from haunted_frog.corpus import DEFAULT_CORPUS

PREFIXES = ([], ["THE"], ["THE", "GHOST"])

for epochs in (1, 20):
    frog, _ = trainer.train(DEFAULT_CORPUS, unit="word", epochs=epochs)
    print(f"=== word-level, {epochs} epoch(s), top_k_edges=16")
    for prefix in PREFIXES:
        top = sorted(frog.probabilities(prefix).items(), key=lambda kv: -kv[1])[:5]
        shown = ", ".join(f"{tok} {p:.3f}" for tok, p in top)
        print(f"  after {' '.join(prefix) or '(start)':<10} -> {shown}")
    print()

# CSE-Frog's own score breakdown (its column labels are Japanese:
# 直接=direct link, 履歴=history, 並び=pair context, 痕跡=trace, 合計点=total, 確率=probability)
frog, _ = trainer.train(DEFAULT_CORPUS, unit="word", epochs=20)
frog.show(["THE"])
