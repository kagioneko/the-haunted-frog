import io
import json

import pytest
from cse import Frog

from haunted_frog import evaluator, trainer
from haunted_frog.corpus import DEFAULT_CORPUS, read_lines
from haunted_frog.generator import Utterance, append_log, generate, generate_many
from haunted_frog.terminal import haunt


@pytest.fixture(scope="module")
def word_frog():
    return trainer.train(DEFAULT_CORPUS, unit="word", epochs=5)


@pytest.fixture(scope="module")
def char_frog():
    return trainer.train(DEFAULT_CORPUS, unit="char", epochs=5)


def test_respects_max_len(word_frog, char_frog):
    frog, _ = word_frog
    for s in range(20):
        u = generate(frog, "word", seed=s, max_len=2)
        assert len(u.tokens) <= 2
    frog, _ = char_frog
    for s in range(20):
        assert len(generate(frog, "char", seed=s, max_len=5).tokens) <= 5


def test_same_seed_same_output(word_frog):
    frog, _ = word_frog
    a = [u.text for u in generate_many(frog, "word", range(10))]
    b = [u.text for u in generate_many(frog, "word", range(10))]
    assert a == b
    assert len(set(a)) > 1          # different seeds do differ


def test_output_is_real_model_output(word_frog):
    # The text shown is exactly the frog's tokens: no filtering, no canned lines.
    frog, _ = word_frog
    u = generate(frog, "word", seed=3)
    assert u.text == " ".join(frog.generate([], n=u.max_len, seed=3))


def test_end_marker_is_not_printed(word_frog, char_frog):
    for (frog, _), unit in ((word_frog, "word"), (char_frog, "char")):
        for u in generate_many(frog, unit, range(30)):
            assert "<END>" not in u.text
            assert all(isinstance(t, str) for t in u.tokens)


def test_ended_means_end_was_possible_at_the_stop(word_frog):
    # Independent of the length rule: wherever we say the frog ended, <END> must
    # actually have had probability at that point.
    frog, _ = word_frog
    ended = [u for u in generate_many(frog, "word", range(50), max_len=50) if u.ended]
    assert ended
    for u in ended:
        probs = frog.probabilities(u.tokens)
        assert any(str(k) == "<END>" and p > 0 for k, p in probs.items())


def test_cut_at_max_len_is_not_ended():
    # A frog that has only seen one long line cannot stop after 1 word.
    frog = Frog().learn([["ONE", "TWO", "THREE", "FOUR", "FIVE"]], epochs=5)
    u = generate(frog, "word", seed=0, max_len=1, prompt="ONE")
    assert u.tokens == ["TWO"] and u.ended is False


def test_untrained_silence_is_an_end_pick():
    probs = Frog().probabilities([])
    assert [str(k) for k in probs] == ["<END>"] and list(probs.values()) == [1.0]


def test_prompt_continuation_and_unknown_prompt(word_frog):
    frog, _ = word_frog
    u = generate(frog, "word", seed=1, prompt="the  ghost")
    # the prompt is normalised and continued, and the prompt itself is not echoed
    assert u.tokens == [str(t) for t in frog.generate(["THE", "GHOST"], n=u.max_len, seed=1)]
    with pytest.raises(ValueError, match="never seen"):
        generate(frog, "word", seed=1, prompt="TAXES")


def test_invalid_max_len(word_frog):
    frog, _ = word_frog
    with pytest.raises(ValueError):
        generate(frog, "word", max_len=0)


def test_untrained_frog_is_silent_not_crashing():
    u = generate(Frog(), "word", seed=0)
    assert u.text == "" and u.tokens == [] and u.ended


def test_log_is_jsonl(tmp_path, word_frog):
    frog, _ = word_frog
    path = tmp_path / "log.jsonl"
    append_log(path, generate_many(frog, "word", range(3)), {"epochs": 5})
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    assert [r["seed"] for r in rows] == [0, 1, 2]
    assert all(r["epochs"] == 5 for r in rows)


def test_haunt_prints_frog_lines_with_seeds(word_frog, tmp_path):
    frog, info = word_frog
    out = io.StringIO()
    spoken = haunt(frog, info, count=3, interval=0, typing_delay=0, seed=100,
                   stream=out, log_path=tmp_path / "h.jsonl")
    text = out.getvalue()
    assert [u.seed for u in spoken] == [100, 101, 102]
    for u in spoken:
        shown = u.text if u.text.strip() else "(silence)"
        assert f"🐸 {shown}" in text and f"[seed {u.seed}]" in text
    assert "\033[" not in text       # not a TTY -> no colour codes


def test_haunt_logs_each_line_as_spoken(word_frog, tmp_path, monkeypatch):
    from haunted_frog import terminal
    frog, info = word_frog
    log = tmp_path / "h.jsonl"
    seen = []
    real = terminal.typewrite

    def spy(text, stream, delay):
        if "[seed" in text:          # by the time a line is printed, it is already logged
            seen.append(len(log.read_text().splitlines()))
        return real(text, stream, delay)

    monkeypatch.setattr(terminal, "typewrite", spy)
    haunt(frog, info, count=3, interval=0, typing_delay=0, seed=5, stream=io.StringIO(),
          log_path=log)
    assert seen == [1, 2, 3]


def test_haunt_rejects_negative_count(word_frog):
    frog, info = word_frog
    with pytest.raises(ValueError):
        haunt(frog, info, count=-1, stream=io.StringIO())


def test_haunt_plain_mode(word_frog):
    frog, info = word_frog
    out = io.StringIO()
    haunt(frog, info, count=2, interval=0, typing_delay=0, seed=1, halloween=False, stream=out)
    assert "🐸" not in out.getvalue() and "> " in out.getvalue()


def test_metrics_on_handmade_utterances():
    lex = {"GHOST", "BOO"}
    lines = ["THE GHOST IS HERE", "BOO"]
    utts = [Utterance("THE GHOST IS HERE", [], "word", 0, "", 12, True),
            Utterance("BOO BOO", [], "word", 1, "", 12, True),
            Utterance("", [], "word", 2, "", 12, True)]
    m = evaluator.metrics(utts, lines, lex)
    assert m["empty_rate"] == round(1 / 3, 4)
    assert m["halloween_word_rate"] == round(3 / 6, 4)
    assert m["copied_corpus_line_rate"] == 0.5
    assert m["novel_line_rate"] == 0.5
    assert m["immediate_repeat_rate"] == round(1 / 4, 4)
    assert m["real_word_rate"] == 1.0


def test_experiment_runs_small(tmp_path):
    result = evaluator.run_experiment(seeds=range(3))
    names = [c["name"] for c in result["conditions"]]
    assert "control-word" in names and "halloween-word-e5" in names
    json_path, md_path = evaluator.write(result, tmp_path / "r.json")
    assert json.loads(json_path.read_text())["n_seeds"] == 3
    assert "| Condition |" in md_path.read_text()
    assert result["untrained"]["metrics"]["empty_rate"] == 1.0


def test_control_corpus_has_no_halloween_words():
    lex = evaluator.read_lexicon()
    words = {w for line in read_lines(evaluator.CONTROL_CORPUS) for w in line.split()}
    assert not (words & lex)


@pytest.mark.parametrize("argv", [
    ["generate", "-n", "-1"], ["generate", "--max-len", "0"], ["haunt", "--count", "-2"],
    ["haunt", "--interval", "nan"], ["evaluate", "--seeds", "0"], ["train", "--epochs", "0"],
])
def test_cli_rejects_bad_numbers(argv):
    from haunted_frog.__main__ import main
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2


def test_cli_reports_missing_model(tmp_path, capsys):
    from haunted_frog.__main__ import main
    assert main(["generate", "--model-dir", str(tmp_path)]) == 1
    assert "haunted_frog train" in capsys.readouterr().err


def test_experiment_needs_seeds():
    with pytest.raises(ValueError):
        evaluator.run_experiment(seeds=[])
