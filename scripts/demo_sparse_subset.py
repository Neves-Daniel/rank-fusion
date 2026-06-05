"""Demo do recuperador esparso num SUBCONJUNTO pequeno do Eurlex-4K.

Objetivo: ver a saída (labels rankeados por query) sem indexar os 130 MB inteiros.
Reusa as funções de src/retrieve_sparse.py; só fatia treino/teste para algo leve.

Uso:
    python -m scripts.demo_sparse_subset            # 300 docs treino, 20 queries
    python -m scripts.demo_sparse_subset 500 30     # tamanhos customizados
"""
from __future__ import annotations

import sys

from src.data import load_dataset
from src.retrieve_sparse import (
    SparseConfig,
    _train_label_columns,
    build_index,
    head_tail_split,
    retrieve,
    write_trec,
)

N_TRAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 300
N_TEST = int(sys.argv[2]) if len(sys.argv) > 2 else 20
N_SHOW = 3          # quantas queries detalhar na tela
TOP_K = 8           # quantos labels mostrar por query


def main() -> None:
    cfg = SparseConfig(
        out_path="data/eurlex4k/runs/sparse_demo.trec",
        cutoff=50,
        num_labels=16,   # menor que o default (64) para o demo
    )

    ds = load_dataset(cfg.raw_dir)
    vocab = ds.label_vocab                      # coluna i -> texto EuroVoc
    all_cols, n_labels = _train_label_columns(cfg.raw_dir)

    train_texts = ds.train.texts[:N_TRAIN]
    train_cols = all_cols[:N_TRAIN]             # rótulos dos MESMOS docs indexados
    test_texts = ds.test.texts[:N_TEST]

    head, tail = head_tail_split(train_cols, n_labels, cfg.head_frac)
    print(f"Subconjunto: {len(train_texts)} docs treino | {len(test_texts)} queries")
    print(f"Rótulos: {n_labels} (cabeça {len(head)} / cauda {len(tail)}) | "
          f"cutoff {cfg.cutoff} | num_labels {cfg.num_labels}\n")

    sr = build_index(train_texts, cfg)
    runs = retrieve(sr, test_texts, train_cols, head, tail, cfg)
    write_trec(runs, cfg.out_path, cfg.tag)
    print(f"run TREC salvo em {cfg.out_path} ({len(runs)} queries)\n")

    # Mostra, para as primeiras N_SHOW queries, os labels rankeados (texto legível)
    for qid in list(runs.keys())[:N_SHOW]:
        ranked = sorted(runs[qid], key=lambda kv: kv[1], reverse=True)[:TOP_K]
        gold = set(ds.test.labels[int(qid)])
        print(f"── Query {qid} ──")
        print(f"   gold ({len(gold)}): {sorted(gold)[:6]}{' ...' if len(gold) > 6 else ''}")
        print(f"   top-{TOP_K} esparso:")
        for rank, (col, score) in enumerate(ranked, 1):
            text = vocab[col]
            faixa = "cauda" if col in tail else "cabeça"
            hit = "✓" if text in gold else " "
            print(f"     {rank:>2}. [{hit}] {text:<35} score={score:7.3f}  ({faixa})")
        print()


if __name__ == "__main__":
    main()
