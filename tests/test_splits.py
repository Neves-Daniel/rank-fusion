"""Testes das funções de src/splits.py (k-fold CV sobre o dataset agrupado).

Puros e leves: make_folds é só numpy; load_pooled escreve um dataset sintético
minúsculo em tmp_path (não toca nos 130 MB reais nem importa retriv).
"""
import numpy as np
import pytest
import scipy.sparse as sp

from src.splits import Fold, PooledData, load_pooled, make_folds


# ----------------------------- make_folds -----------------------------

def test_make_folds_particiona_todos_os_docs_sem_sobreposicao():
    folds = make_folds(20, k=5, seed=42)
    assert len(folds) == 5
    # cada doc é teste de exatamente um fold
    todos_test = np.concatenate([f.test_idx for f in folds])
    assert sorted(todos_test.tolist()) == list(range(20))
    # 20/5 = 4 queries por fold
    assert all(len(f.test_idx) == 4 for f in folds)


def test_make_folds_train_e_test_sao_complementares_e_disjuntos():
    n = 23
    for fold in make_folds(n, k=5, seed=7):
        assert set(fold.train_idx.tolist()).isdisjoint(fold.test_idx.tolist())
        assert sorted(fold.train_idx.tolist() + fold.test_idx.tolist()) == list(range(n))
        # índices vêm ordenados (determinístico)
        assert list(fold.train_idx) == sorted(fold.train_idx)
        assert list(fold.test_idx) == sorted(fold.test_idx)


def test_make_folds_e_deterministico_por_seed():
    a = make_folds(50, k=5, seed=123)
    b = make_folds(50, k=5, seed=123)
    c = make_folds(50, k=5, seed=999)
    assert [f.test_idx.tolist() for f in a] == [f.test_idx.tolist() for f in b]
    assert [f.test_idx.tolist() for f in a] != [f.test_idx.tolist() for f in c]


def test_make_folds_tamanhos_quase_iguais_quando_nao_divide():
    # 22 docs em 5 folds -> tamanhos 5,5,4,4,4
    sizes = sorted(len(f.test_idx) for f in make_folds(22, k=5, seed=0))
    assert sizes == [4, 4, 4, 5, 5]


def test_make_folds_rejeita_k_invalido():
    with pytest.raises(ValueError, match="k >= 2"):
        make_folds(10, k=1)


def test_make_folds_rejeita_n_menor_que_k():
    with pytest.raises(ValueError, match="menor que k"):
        make_folds(3, k=5)


# ----------------------------- load_pooled -----------------------------

def _write_dataset(raw_dir, vocab, trn_texts, tst_texts, Ytrn, Ytst):
    (raw_dir / "Y.txt").write_text("\n".join(vocab), encoding="utf-8")
    (raw_dir / "trn_X.txt").write_text("\n".join(trn_texts), encoding="utf-8")
    (raw_dir / "tst_X.txt").write_text("\n".join(tst_texts), encoding="utf-8")
    sp.save_npz(str(raw_dir / "Y.trn.npz"), sp.csr_matrix(Ytrn))
    sp.save_npz(str(raw_dir / "Y.tst.npz"), sp.csr_matrix(Ytst))


def test_load_pooled_concatena_treino_e_teste_na_ordem_global(tmp_path):
    vocab = ["a", "b", "c"]
    Ytrn = np.array([[1, 1, 0], [0, 0, 1]])      # 2 docs treino
    Ytst = np.array([[1, 0, 1]])                  # 1 doc teste
    _write_dataset(tmp_path, vocab, ["treino0", "treino1"], ["teste0"], Ytrn, Ytst)

    pooled = load_pooled(str(tmp_path))

    assert isinstance(pooled, PooledData)
    assert len(pooled) == 3
    assert pooled.n_labels == 3
    # ordem global = [treino..., teste...]
    assert pooled.texts == ["treino0", "treino1", "teste0"]
    # label_cols por índice global (colunas não-nulas de cada linha)
    assert pooled.label_cols == [[0, 1], [2], [0, 2]]


def test_load_pooled_rejeita_descasamento_texto_matriz(tmp_path):
    vocab = ["a", "b"]
    Ytrn = np.array([[1, 0], [0, 1]])             # 2 linhas...
    Ytst = np.array([[1, 1]])
    _write_dataset(tmp_path, vocab, ["so um texto de treino"], ["teste0"], Ytrn, Ytst)
    with pytest.raises(ValueError, match="treino"):
        load_pooled(str(tmp_path))


def test_load_pooled_rejeita_colunas_diferentes_entre_trn_e_tst(tmp_path):
    vocab = ["a", "b"]
    Ytrn = np.array([[1, 0]])                      # 2 colunas
    Ytst = np.array([[1, 0, 1]])                   # 3 colunas
    _write_dataset(tmp_path, vocab, ["treino0"], ["teste0"], Ytrn, Ytst)
    with pytest.raises(ValueError, match="colunas"):
        load_pooled(str(tmp_path))


def test_folds_sobre_pooled_cobrem_todo_o_dataset(tmp_path):
    # integração leve: load_pooled + make_folds geram cobertura completa
    vocab = ["a", "b"]
    Ytrn = np.eye(8, 2, dtype=int)[:, :2] if False else np.array([[i % 2, (i + 1) % 2] for i in range(8)])
    Ytst = np.array([[1, 1], [0, 1]])
    _write_dataset(tmp_path, vocab, [f"trn{i}" for i in range(8)], ["tst0", "tst1"], Ytrn, Ytst)

    pooled = load_pooled(str(tmp_path))
    folds = make_folds(len(pooled), k=5, seed=42)
    cobertos = np.concatenate([f.test_idx for f in folds])
    assert sorted(cobertos.tolist()) == list(range(10))
