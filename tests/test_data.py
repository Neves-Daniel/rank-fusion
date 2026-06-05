"""Testes de src/data.py com dados sintéticos minúsculos.

Nada aqui carrega os arquivos reais (130 MB) nem depende de download: cada teste
cria um dataset de 2-3 documentos em tmp_path. Roda em <1 s e não tem como
estourar memória.
"""
import numpy as np
import pytest
import scipy.sparse as sp

from src.data import (
    Dataset,
    Split,
    _labels_from_matrix,
    dataset_stats,
    load_dataset,
)


def _write_dataset(raw_dir, vocab, trn_texts, tst_texts, Ytrn, Ytst):
    """Escreve um dataset sintético no layout esperado por load_dataset."""
    (raw_dir / "Y.txt").write_text("\n".join(vocab), encoding="utf-8")
    (raw_dir / "trn_X.txt").write_text("\n".join(trn_texts), encoding="utf-8")
    (raw_dir / "tst_X.txt").write_text("\n".join(tst_texts), encoding="utf-8")
    sp.save_npz(str(raw_dir / "Y.trn.npz"), sp.csr_matrix(Ytrn))
    sp.save_npz(str(raw_dir / "Y.tst.npz"), sp.csr_matrix(Ytst))


def test_labels_from_matrix_maps_columns_to_vocab():
    vocab = ["esporte", "política", "saúde"]
    # doc0 -> colunas {0,2}; doc1 -> coluna {1}; doc2 -> nenhuma
    Y = sp.csr_matrix(np.array([[1, 0, 1], [0, 1, 0], [0, 0, 0]]))
    out = _labels_from_matrix(Y, vocab)
    assert out == [["esporte", "saúde"], ["política"], []]


def test_labels_from_matrix_rejects_vocab_mismatch():
    Y = sp.csr_matrix(np.zeros((2, 3)))
    with pytest.raises(ValueError, match="colunas"):
        _labels_from_matrix(Y, ["a", "b"])  # 2 rótulos para 3 colunas


def test_load_dataset_roundtrip(tmp_path):
    vocab = ["a", "b", "c"]
    Ytrn = np.array([[1, 1, 0], [0, 0, 1]])      # 2 docs treino
    Ytst = np.array([[1, 0, 1]])                  # 1 doc teste
    _write_dataset(tmp_path, vocab, ["doc treino um", "doc treino dois"],
                   ["doc teste"], Ytrn, Ytst)

    ds = load_dataset(str(tmp_path))

    assert isinstance(ds, Dataset)
    assert len(ds.train) == 2 and len(ds.test) == 1
    assert ds.label_vocab == vocab
    assert ds.train.texts[0] == "doc treino um"
    assert ds.train.labels == [["a", "b"], ["c"]]
    assert ds.test.labels == [["a", "c"]]


def test_load_dataset_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_eurlex"):
        load_dataset(str(tmp_path))  # diretório vazio: falta Y.txt


def test_load_dataset_text_row_mismatch(tmp_path):
    vocab = ["a", "b", "c"]
    Ytrn = np.array([[1, 0, 0], [0, 1, 0]])       # 2 linhas em Y...
    Ytst = np.array([[0, 0, 1]])
    _write_dataset(tmp_path, vocab, ["um unico doc de treino"],  # ...mas 1 texto
                   ["doc teste"], Ytrn, Ytst)
    with pytest.raises(ValueError, match="treino"):
        load_dataset(str(tmp_path))


def test_dataset_stats_counts():
    train = Split(["t0", "t1", "t2"], [["a", "b"], ["a"], ["a", "b", "c"]], "train")
    test = Split(["s0"], [["d"]], "test")
    ds = Dataset(train=train, test=test, label_vocab=["a", "b", "c", "d", "e"])

    stats = dataset_stats(ds)

    assert stats["n_train"] == 3
    assert stats["n_test"] == 1
    assert stats["n_labels_vocab"] == 5
    assert stats["n_labels_used"] == 4          # a,b,c (treino) + d (teste); 'e' nunca usado
    assert stats["avg_labels_per_doc"] == pytest.approx((2 + 1 + 3) / 3, abs=1e-3)
    assert stats["max_labels_per_doc"] == 3
