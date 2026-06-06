"""Splits de validação cruzada (k-fold) sobre o dataset XMTC AGRUPADO (treino+teste).

Por que agrupar treino+teste: o artigo-base (França et al., 2025, "Ranking-based
Fusion Algorithms for XMTC", arXiv 2507.03761) avalia com **5-fold cross-validation
sobre todo o dataset** (N instâncias), e não sobre o split fixo treino/teste do
PECOS — "we adopt a 5-fold cross-validation approach across all datasets ...
averaged across the five test splits ... mitigates the risk of dataset-specific
biases inherent in single train-test splits". Em cada fold, um doc é query (teste)
exatamente UMA vez; os 4/5 restantes formam o corpus indexado naquele fold.

Para o Eurlex-4K isso bate com a Tabela 1 do artigo: N = 19.314 = 15.449 (treino
PECOS) + 3.865 (teste PECOS); 19.314/5 ≈ 3.863 queries por fold.

IDs globais: os documentos são indexados 0..N-1 na ordem [treino PECOS, depois
teste PECOS]. Esse índice global é a chave do doc (qid no run TREC, e linha em
`label_cols`), estável entre folds — o gold de uma query é `label_cols[qid]`.

Reprodutibilidade: a atribuição de folds é determinística dada a `seed` (numpy
RandomState). A seed dos folds do artigo é desconhecida; reproduzimos a
METODOLOGIA (5-fold CV, mesmas definições), não os índices exatos deles.

A definição de cabeça/cauda (Pareto 80/20) é uma propriedade GLOBAL do dataset no
artigo (frequências sobre os N documentos), não por fold — calcule-a uma vez sobre
`PooledData.label_cols`, fora deste módulo (ver retrieve_sparse).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from src.data import _read_lines


@dataclass
class PooledData:
    """Dataset agrupado (treino+teste) com índices globais 0..N-1."""

    texts: list[str]                 # texto[i] do doc global i
    label_cols: list[list[int]]      # colunas (índices) dos rótulos-gold do doc global i
    n_labels: int                    # nº de colunas de Y (vocabulário de rótulos)

    def __len__(self) -> int:
        return len(self.texts)


@dataclass
class Fold:
    """Um fold de CV: índices globais do corpus (treino) e das queries (teste)."""

    fold_id: int
    train_idx: np.ndarray            # índices globais indexados como corpus neste fold
    test_idx: np.ndarray             # índices globais usados como query neste fold


def load_pooled(raw_dir: str) -> PooledData:
    """Carrega treino+teste do PECOS e os agrupa num único conjunto indexado 0..N-1.

    Ordem global: primeiro os docs de treino (0..n_trn-1), depois os de teste
    (n_trn..n_trn+n_tst-1). Usa as matrizes Y direto (em vez dos IDs textuais de
    data.py) porque o índice de coluna é a chave de rótulo dos runs.
    """
    trn_texts = _read_lines(os.path.join(raw_dir, "trn_X.txt"))
    tst_texts = _read_lines(os.path.join(raw_dir, "tst_X.txt"))

    Ytrn = sp.load_npz(os.path.join(raw_dir, "Y.trn.npz")).tocsr()
    Ytst = sp.load_npz(os.path.join(raw_dir, "Y.tst.npz")).tocsr()

    if Ytrn.shape[0] != len(trn_texts):
        raise ValueError(f"treino: {len(trn_texts)} textos vs {Ytrn.shape[0]} linhas em Y.trn")
    if Ytst.shape[0] != len(tst_texts):
        raise ValueError(f"teste: {len(tst_texts)} textos vs {Ytst.shape[0]} linhas em Y.tst")
    if Ytrn.shape[1] != Ytst.shape[1]:
        raise ValueError(f"Y.trn tem {Ytrn.shape[1]} colunas mas Y.tst tem {Ytst.shape[1]}.")

    Y = sp.vstack([Ytrn, Ytst]).tocsr()
    texts = trn_texts + tst_texts
    label_cols = [
        Y.indices[Y.indptr[i]:Y.indptr[i + 1]].tolist()
        for i in range(Y.shape[0])
    ]
    return PooledData(texts=texts, label_cols=label_cols, n_labels=Y.shape[1])


def make_folds(n_docs: int, k: int = 5, seed: int = 42) -> list[Fold]:
    """Particiona 0..n_docs-1 em `k` folds de CV (embaralhamento determinístico).

    Cada doc cai no teste de exatamente um fold; os demais formam o treino daquele
    fold. Índices retornados em ordem crescente (determinístico, fácil de inspecionar).
    """
    if k < 2:
        raise ValueError(f"k-fold precisa de k >= 2, recebido k={k}.")
    if n_docs < k:
        raise ValueError(f"n_docs={n_docs} menor que k={k}: folds vazios.")

    rng = np.random.RandomState(seed)
    perm = rng.permutation(n_docs)
    chunks = np.array_split(perm, k)            # k blocos quase iguais

    folds: list[Fold] = []
    for f in range(k):
        test_idx = np.sort(chunks[f])
        train_idx = np.sort(
            np.concatenate([chunks[j] for j in range(k) if j != f])
        )
        folds.append(Fold(fold_id=f, train_idx=train_idx, test_idx=test_idx))
    return folds


if __name__ == "__main__":
    import sys

    raw = sys.argv[1] if len(sys.argv) > 1 else "data/eurlex4k/raw"
    pooled = load_pooled(raw)
    folds = make_folds(len(pooled), k=5)
    print(f"Diretório: {raw}")
    print(f"  docs agrupados (N)     : {len(pooled)}")
    print(f"  rótulos (colunas de Y) : {pooled.n_labels}")
    print(f"  folds                  : {len(folds)} (seed=42)")
    for fold in folds:
        print(f"    fold {fold.fold_id}: corpus={len(fold.train_idx)} | queries={len(fold.test_idx)}")
