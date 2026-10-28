"""The Haunted Terminal.

Two kinds of text appear on screen, and they are drawn differently on purpose:

  * narration  (dim, no frog)   - fixed, hand-written flavour text
  * 🐸 lines    (bright green)   - real CSE-Frog output, printed verbatim

The seed of every frog line is shown next to it so any line can be reproduced.
"""
from __future__ import annotations

import os
import random
import sys
import time

from .generator import append_log, generate

FROG_ART = r"""
        _   _
       (o)-(o)
    .-(   "   )-.
   /   '-----'   \
   \_/ \_____/ \_/
"""

BANNER = "[ HAUNTED FROG TERMINAL ]"

# Scripted narration. Never presented as model output.
NARRATION_OPEN = [
    "A strange presence has awakened...",
    "Something small is breathing inside the machine.",
]
NARRATION_BETWEEN = [
    "...the cursor blinks on its own...",
    "...a faint croak in the fan noise...",
    "...the screen gets a little colder...",
    "...",
]
NARRATION_CLOSE = "The frog is still watching."


class Style:
    def __init__(self, enabled: bool, halloween: bool):
        on = enabled and halloween
        self.frog = "\033[1;92m" if on else ""
        self.dim = "\033[2;37m" if on else ""
        self.ghost = "\033[38;5;208m" if on else ""
        self.reset = "\033[0m" if on else ""
        self.clear = "\033[2J\033[H" if on else ""
        self.halloween = halloween


def color_enabled(stream) -> bool:
    return stream.isatty() and "NO_COLOR" not in os.environ


def typewrite(text, stream, delay: float):
    for ch in text:
        stream.write(ch)
        stream.flush()
        if delay:
            time.sleep(delay)
    stream.write("\n")
    stream.flush()


def haunt(frog, info, *, count: int = 8, interval: float = 2.5, seed=None,
          typing_delay: float = 0.05, halloween: bool = True, prompt: str = "",
          log_path=None, stream=None):
    """Let the frog speak `count` times (0 = until Ctrl+C).

    `seed` fixes the whole session: line i uses seed+i, so a session can be replayed.
    """
    stream = stream or sys.stdout
    style = Style(color_enabled(stream), halloween)
    base = seed if seed is not None else random.SystemRandom().randrange(1_000_000)
    narrate = random.Random(base)          # narration choice only; never touches the model
    unit = info.unit

    stream.write(style.clear)
    if halloween:
        stream.write(f"{style.ghost}{BANNER}{style.reset}\n{style.frog}{FROG_ART}{style.reset}\n")
        typewrite(f"{style.dim}{narrate.choice(NARRATION_OPEN)}{style.reset}", stream, typing_delay)
    else:
        stream.write(f"CSE-Frog generator ({unit}-level, {info.epochs} epochs)\n")
    stream.write(f"{style.dim}(model: cse-frog {info.cse_frog_version}, {unit}-level, "
                 f"{info.lines} lines x {info.epochs} epochs, session seed {base}){style.reset}\n\n")

    spoken = []
    i = 0
    try:
        while count == 0 or i < count:
            utt = generate(frog, unit, seed=base + i, prompt=prompt)
            spoken.append(utt)
            text = utt.text if utt.text.strip() else "(silence)"
            mark = "" if utt.ended else " …"
            prefix = "🐸 " if halloween else "> "
            typewrite(f"{style.frog}{prefix}{text}{mark}{style.reset}"
                      f"{style.dim}   [seed {utt.seed}]{style.reset}", stream, typing_delay)
            i += 1
            if count == 0 or i < count:
                if halloween and narrate.random() < 0.35:
                    stream.write(f"{style.dim}{narrate.choice(NARRATION_BETWEEN)}{style.reset}\n")
                time.sleep(interval)
    except KeyboardInterrupt:
        stream.write("\n")
    if halloween:
        typewrite(f"\n{style.dim}{NARRATION_CLOSE}{style.reset}", stream, typing_delay)
    if log_path:
        append_log(log_path, spoken, {"session_seed": base, "epochs": info.epochs})
    return spoken
