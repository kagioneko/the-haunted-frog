"""Train both frogs, then open a short Haunted Terminal session.

    python scripts/run_demo.py
"""
from haunted_frog.__main__ import main

if __name__ == "__main__":
    for unit in ("word", "char"):
        main(["train", "--unit", unit])
    raise SystemExit(main(["haunt", "--count", "6", "--seed", "1003"]))
