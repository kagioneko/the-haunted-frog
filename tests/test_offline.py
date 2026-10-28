"""Training and generation must work with the network switched off."""
import socket

import pytest

from haunted_frog import trainer
from haunted_frog.corpus import DEFAULT_CORPUS
from haunted_frog.generator import generate_many


@pytest.fixture
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("network access attempted")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


def test_train_generate_save_load_offline(no_network, tmp_path):
    frog, info = trainer.train(DEFAULT_CORPUS, unit="word", epochs=2)
    trainer.save(frog, info, tmp_path)
    loaded, _ = trainer.load(tmp_path, "word")
    assert len(generate_many(loaded, "word", range(5))) == 5


def test_no_network_libraries_imported():
    # Fresh interpreter, so pytest plugins cannot pollute sys.modules.
    import subprocess
    import sys
    code = ("import sys, haunted_frog.__main__; "
            "print([m for m in ('requests', 'urllib3', 'httpx', 'aiohttp', 'http.client') "
            "if m in sys.modules])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
