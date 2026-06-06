"""Demo da geração de RAG-labels num SUBCONJUNTO de rótulos (fold 0 só).

Objetivo: validar o prompt e o backend vLLM gerando poucas descrições, antes de
rodar os ~3.956 × 5 folds. Imprime, para alguns rótulos, o nome e a descrição
gerada para conferir na mão. Reusa as funções de src/label_desc.py.

Requer vLLM + GPU (roda na Brev). Mira de propósito os primeiros rótulos, que no
Eurlex-4K incluem os códigos numéricos do EuroVoc (caso de borda).

Uso:
    python -m scripts.demo_label_desc_subset            # 20 rótulos, fold 0
    python -m scripts.demo_label_desc_subset 30         # N rótulos
"""
from __future__ import annotations

import os
import sys

from src.data import _read_lines
from src.label_desc import LabelDescConfig, generate_fold, make_backend
from src.splits import load_pooled, make_folds

N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
N_SHOW = 3


def main() -> None:
    cfg = LabelDescConfig(
        out_dir="data/eurlex4k/rag-labels-demo",   # separado dos artefatos oficiais
        label_subset=N,
        batch_size=8,
    )
    pooled = load_pooled(cfg.raw_dir)
    vocab = _read_lines(os.path.join(cfg.raw_dir, "Y.txt"))
    print(f"Subconjunto: {N} rótulos (de {pooled.n_labels}) | fold 0 | {cfg.model}\n")

    backend = make_backend(cfg)
    fold0 = make_folds(len(pooled), k=cfg.n_folds, seed=cfg.seed)[0]
    res = generate_fold(fold0, pooled, vocab, backend, cfg)

    print()
    for col in list(res.keys())[:N_SHOW]:
        print(f"── rótulo {col}: {vocab[col]!r} ──")
        print(res[col][:600])
        print()


if __name__ == "__main__":
    main()
