import json
import math
import numpy as np

from edugpt.core import tokenize, prepare_corpus
from edugpt.trainer import make_pairs, neighbors, train


def test_spanish_tokenization():
    assert tokenize("¡Niños y NIÑAS! Árbol-azul 2026.") == ["niños", "y", "niñas", "árbol-azul", "2026"]


def test_corpus_vocabulary_and_sources(tmp_path):
    books = tmp_path / "books"
    books.mkdir()
    (books / "uno.txt").write_text("perro gato perro mascota animal gato", encoding="utf-8")
    (books / "dos.txt").write_text("gato animal perro animal perro", encoding="utf-8")
    corpus = prepare_corpus(books, tmp_path / "corpus.json", min_freq=2)
    assert corpus["vocab"]["<unk>"] == 0
    assert corpus["stats"]["books"] == 2
    assert len(corpus["documents"]) == 2
    assert json.loads((tmp_path / "corpus.json").read_text(encoding="utf-8"))["stats"]["tokens"] == 11


def test_no_cross_document_pairs():
    pairs = make_pairs([{"ids": [1, 2]}, {"ids": [3, 4]}], window=2, max_pairs=100, seed=3)
    assert not any(a in (1, 2) and b in (3, 4) or a in (3, 4) and b in (1, 2) for a, b in pairs)


def test_training_creates_real_artifacts(tmp_path):
    books = tmp_path / "books"
    books.mkdir()
    (books / "base.txt").write_text("el perro corre el gato juega el perro juega el gato corre " * 15, encoding="utf-8")
    corpus = prepare_corpus(books, tmp_path / "corpus.json")
    run = tmp_path / "run"
    metrics = train(corpus, run, dimensions=8, window=2, epochs=2,
                    negatives=2, batch_size=32, max_pairs=400, seed=42)
    assert len(metrics) == 2
    assert all(math.isfinite(row["loss"]) for row in metrics)
    original = np.load(run / "embeddings_initial.npy")
    final = np.load(run / "embeddings_final.npy")
    assert not np.array_equal(original, final)
    assert (run / "model.pt").exists()
    assert neighbors(final, corpus["vocab"], "perro", 3)
