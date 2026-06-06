"""Demo do recuperador denso num SUBCONJUNTO pequeno do Eurlex-4K.

Objetivo: ver o pipeline denso ponta a ponta (treino contrastivo curto +
inferência → run TREC) sem rodar os 5 folds inteiros. Reusa as funções de
src/retrieve_dense.py; só fatia o dataset agrupado para algo leve.

REQUER GPU (Brev) + `pip install pytorch-metric-learning`. NÃO rodar na WSL local
(treina BERT de verdade; ver memória de WSL/OOM).

Uso (na Brev, dentro do container):
    python -m scripts.demo_dense_subset                 # 300 docs treino, 20 queries
    python -m scripts.demo_dense_subset 500 30          # tamanhos customizados
    python -m scripts.demo_dense_subset 300 20 NONE     # sem RAG-labels (nome cru)
"""
from __future__ import annotations

import os
import sys

import numpy as np

from src.data import _read_lines
from src.retrieve_dense import (
    DenseConfig,
    build_relevance_map,
    infer_fold,
    load_fold_descriptions,
    resolve_label_texts,
    train_fold,
)
from src.retrieve_sparse import head_tail_split, write_trec
from src.splits import Fold, PooledData, load_pooled

N_TRAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 300
N_TEST = int(sys.argv[2]) if len(sys.argv) > 2 else 20
ENHANCEMENT = sys.argv[3] if len(sys.argv) > 3 else "LLM"
N_SHOW = 3          # quantas queries detalhar na tela
TOP_K = 8           # quantos labels mostrar por query


def main() -> None:
    cfg = DenseConfig(
        epochs=1,                 # demo: 1 época só
        batch_size=16,
        num_labels=16,            # menor que o default (64) para o demo
        encode_batch_size=64,
        label_enhancement=ENHANCEMENT,
        out_template="dense_demo.fold{fold}.trec",
    )

    pooled_full = load_pooled(cfg.raw_dir)
    vocab = _read_lines(os.path.join(cfg.raw_dir, "Y.txt"))

    m = N_TRAIN + N_TEST
    sub = PooledData(
        texts=pooled_full.texts[:m],
        label_cols=pooled_full.label_cols[:m],
        n_labels=pooled_full.n_labels,
    )
    fold = Fold(
        fold_id=0,
        train_idx=np.arange(N_TRAIN),
        test_idx=np.arange(N_TRAIN, m),
    )

    head, tail = head_tail_split(sub.label_cols, sub.n_labels, cfg.head_frac)
    relevance_map = build_relevance_map(sub.label_cols)
    descriptions = load_fold_descriptions(cfg, fold.fold_id)
    label_texts = resolve_label_texts(vocab, descriptions, cfg.label_enhancement)

    print(f"Subconjunto: {N_TRAIN} docs treino | {N_TEST} queries | "
          f"enhancement={cfg.label_enhancement} ({len(descriptions)} descrições)")
    print(f"Rótulos: {sub.n_labels} (cabeça {len(head)} / cauda {len(tail)}) | "
          f"num_labels {cfg.num_labels} | modelo {cfg.architecture}\n")

    encoder, tokenizer = train_fold(fold, sub, label_texts, relevance_map, cfg)
    runs = infer_fold(encoder, tokenizer, fold, sub, label_texts, head, tail, cfg)

    out_path = cfg.fold_out_path(fold.fold_id)
    write_trec(runs, out_path, cfg.tag)
    print(f"\nrun TREC salvo em {out_path} ({len(runs)} queries)\n")

    # Mostra, para as primeiras N_SHOW queries, os labels rankeados (texto legível)
    for qid in list(runs.keys())[:N_SHOW]:
        ranked = sorted(runs[qid], key=lambda kv: kv[1], reverse=True)[:TOP_K]
        gold = set(sub.label_cols[int(qid)])
        print(f"── Query {qid} ──")
        print(f"   gold ({len(gold)}): {sorted(gold)[:6]}{' ...' if len(gold) > 6 else ''}")
        print(f"   top-{TOP_K} denso:")
        for rank, (col, score) in enumerate(ranked, 1):
            faixa = "cauda" if col in tail else "cabeça"
            hit = "✓" if col in gold else " "
            print(f"     {rank:>2}. [{hit}] {vocab[col]:<35} score={score:7.4f}  ({faixa})")
        print()


if __name__ == "__main__":
    main()
