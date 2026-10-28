import json

import pytest
from cse import Frog

from haunted_frog import trainer
from haunted_frog.corpus import DEFAULT_CORPUS, read_lines, to_sequences


def test_corpus_loads_and_skips_comments():
    lines = read_lines(DEFAULT_CORPUS)
    assert len(lines) == 60
    assert "TRICK OR TREAT" in lines
    assert not any(line.startswith("#") for line in lines)


def test_corpus_normalizes_case_and_spaces(tmp_path):
    p = tmp_path / "c.txt"
    p.write_text("# note\n\n  trick   or treat \nBoo\n", encoding="utf-8")
    assert read_lines(p) == ["TRICK OR TREAT", "BOO"]


def test_empty_corpus_is_an_error(tmp_path):
    p = tmp_path / "c.txt"
    p.write_text("# only comments\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_lines(p)


def test_lines_stay_separate_sequences():
    # A flat list of strings is ONE sequence to CSE-Frog; we must pass a list of lists.
    seqs = to_sequences(["BOO", "BLACK CAT"], "word")
    assert seqs == [["BOO"], ["BLACK", "CAT"]]
    assert to_sequences(["BOO"], "char") == [["B", "O", "O"]]


def test_line_boundaries_are_learned():
    frog, _ = trainer.train(DEFAULT_CORPUS, unit="word", epochs=5)
    # "HALLOWEEN" only ever ends a line, so the frog must expect <END> after it,
    # not glue on the first word of the next line ("TRICK").
    probs = frog.probabilities(["HAPPY", "HALLOWEEN"])
    best = max(probs, key=probs.get)
    assert str(best) == "<END>"


def test_train_records_metadata():
    _, info = trainer.train(DEFAULT_CORPUS, unit="word", epochs=2)
    assert info.lines == 60
    assert info.epochs == 2
    assert info.unit == "word"
    assert info.cse_frog_version == trainer.cse_frog_version()
    assert info.frog_config == trainer.DEFAULT_FROG_CONFIG
    assert len(info.corpus_sha256) == 64


def test_bad_arguments():
    with pytest.raises(ValueError):
        trainer.train(DEFAULT_CORPUS, unit="sentence")
    with pytest.raises(ValueError):
        trainer.train(DEFAULT_CORPUS, epochs=0)


def test_save_and_load_roundtrip(tmp_path):
    frog, info = trainer.train(DEFAULT_CORPUS, unit="word", epochs=3)
    model_path, meta_path = trainer.save(frog, info, tmp_path)
    loaded, loaded_info = trainer.load(tmp_path, "word")
    assert isinstance(loaded, Frog)
    assert loaded_info == info
    assert json.loads(meta_path.read_text())["unit"] == "word"
    for seed in range(5):
        assert loaded.generate([], n=12, seed=seed) == frog.generate([], n=12, seed=seed)


def test_load_missing_model_has_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="haunted_frog train"):
        trainer.load(tmp_path, "word")
