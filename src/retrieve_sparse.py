"""Recuperador esparso (BM25) para XMTC — metodologia xCoRetriev / RAG-Fuse.

Abordagem: kNN léxico ("documentos parecidos têm rótulos parecidos").
  1. Indexa os documentos de TREINO com BM25.
  2. Para cada documento de TESTE (query), recupera os top-`cutoff` vizinhos
     de treino, cada um com sua nota BM25.
  3. Cada vizinho "vota" nos seus próprios rótulos-gold, com peso = nota BM25.
     O score de um rótulo é a SOMA (ou MÁX) das notas dos vizinhos que o têm.
  4. Split cabeça/cauda (Pareto): mantém os top-`num_labels` rótulos de cabeça
     E os top-`num_labels` de cauda (ex.: 64 + 64 = 128 candidatos). É o que dá
     voz à cauda — o "Dynamic Two-Stage" do xCoRetriev.
  5. Exporta um run TREC (`qid Q0 label_id rank score tag`), consumido depois
     por fusion.py / metrics.py.

Parâmetros do BM25 (k1=1.5, b=0.75) e o pré-processamento (word/stemmer/stopwords
+ normalizações) replicam SparseRetriever.py do RAG-Fuse (celsofranssa/RAG-Fuse).

Chave de rótulo no run = `label_{índice_da_coluna em Y}`, como no RAG-Fuse. O texto
EuroVoc correspondente está em Y.txt (coluna i = linha i) e tem espaços, por isso
não serve como token TREC; o índice da coluna é estável dentro do dataset.

NÃO há treino aqui: BM25 é estatístico. (Treino só apareceria no recuperador denso
com fine-tuning, que está fora da v1.)

Uso:
    python src/retrieve_sparse.py            # roda no Eurlex-4K, gera runs/sparse.trec
"""
from __future__ import annotations

import heapq
import os
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from src.data import load_dataset


@dataclass
class SparseConfig:
    raw_dir: str = "data/eurlex4k/raw"
    out_path: str = "data/eurlex4k/runs/sparse.trec"
    cutoff: int = 100          # nº de vizinhos (docs de treino) recuperados por query
    num_labels: int = 64       # nº de rótulos mantidos POR classe (cabeça/cauda)
    aggregation: str = "sum"   # "sum" (CombSUM) ou "max" (CombMAX)
    head_frac: float = 0.20    # 20% rótulos mais frequentes = cabeça (Pareto)
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    tag: str = "bm25"


def _train_label_columns(raw_dir: str) -> tuple[list[list[int]], int]:
    """Para cada doc de treino, as colunas (índices inteiros) dos seus rótulos.

    Usa a matriz Y direto (em vez dos IDs textuais de data.py) porque o esparso
    precisa do índice de coluna, que é a chave de rótulo do run.
    """
    Ytrn = sp.load_npz(os.path.join(raw_dir, "Y.trn.npz")).tocsr()
    cols = [
        Ytrn.indices[Ytrn.indptr[i]:Ytrn.indptr[i + 1]].tolist()
        for i in range(Ytrn.shape[0])
    ]
    return cols, Ytrn.shape[1]


def head_tail_split(
    train_label_cols: list[list[int]], n_labels: int, head_frac: float
) -> tuple[set[int], set[int]]:
    """Particiona os rótulos: os `head_frac` mais frequentes no TREINO = cabeça;
    o restante = cauda (regra de Pareto: 80% menos frequentes = cauda)."""
    freq = np.zeros(n_labels, dtype=np.int64)
    for cols in train_label_cols:
        for c in cols:
            freq[c] += 1
    order = np.argsort(-freq, kind="stable")          # mais frequente primeiro
    n_head = int(round(head_frac * n_labels))
    head = set(order[:n_head].tolist())
    tail = set(order[n_head:].tolist())
    return head, tail


def build_index(train_texts: list[str], cfg: SparseConfig):
    """Indexa os documentos de treino com BM25 (retriv), replicando o RAG-Fuse."""
    from retriv import SparseRetriever

    index_name = "xmtc_sparse"
    try:
        SparseRetriever.delete(index_name)            # idempotência / reprodutibilidade
    except Exception:
        pass

    sr = SparseRetriever(
        index_name=index_name,
        model="bm25",
        min_df=1,
        tokenizer="word",
        stemmer="english",
        stopwords="english",
        do_lowercasing=True,
        do_ampersand_normalization=True,
        do_special_chars_normalization=True,
        do_acronyms_normalization=True,
        do_punctuation_removal=True,
        hyperparams={"k1": cfg.bm25_k1, "b": cfg.bm25_b},
    )
    collection = [{"id": str(i), "text": t} for i, t in enumerate(train_texts)]
    sr.index(collection)
    return sr


def retrieve(
    sr,
    test_texts: list[str],
    train_label_cols: list[list[int]],
    head: set[int],
    tail: set[int],
    cfg: SparseConfig,
) -> dict[str, list[tuple[int, float]]]:
    """Recupera vizinhos e agrega seus rótulos em scores, por classe (cabeça/cauda).

    Retorna {qid: [(coluna_rótulo, score), ...]} com até num_labels*2 candidatos.
    """
    queries = [{"id": str(i), "text": t} for i, t in enumerate(test_texts)]
    results = sr.bsearch(queries=queries, cutoff=cfg.cutoff)  # {qid: {docid: score}}

    runs: dict[str, list[tuple[int, float]]] = {}
    for qid, neighbors in results.items():
        head_scores: dict[int, float] = {}
        tail_scores: dict[int, float] = {}
        for doc_id, score in neighbors.items():
            for c in train_label_cols[int(doc_id)]:
                bucket = head_scores if c in head else tail_scores
                if cfg.aggregation == "sum":
                    bucket[c] = bucket.get(c, 0.0) + score
                elif cfg.aggregation == "max":
                    if score > bucket.get(c, float("-inf")):
                        bucket[c] = score
                else:
                    raise ValueError("aggregation deve ser 'sum' ou 'max'.")

        top_head = heapq.nlargest(cfg.num_labels, head_scores.items(), key=lambda kv: kv[1])
        top_tail = heapq.nlargest(cfg.num_labels, tail_scores.items(), key=lambda kv: kv[1])
        runs[qid] = top_head + top_tail
    return runs


def write_trec(runs: dict[str, list[tuple[int, float]]], out_path: str, tag: str) -> None:
    """Escreve o run no formato TREC: `qid Q0 label_id rank score tag`."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for qid, items in runs.items():
            ranked = sorted(items, key=lambda kv: kv[1], reverse=True)
            for rank, (col, score) in enumerate(ranked, start=1):
                fh.write(f"{qid} Q0 label_{col} {rank} {score:.6f} {tag}\n")


def main(cfg: SparseConfig | None = None) -> None:
    cfg = cfg or SparseConfig()
    ds = load_dataset(cfg.raw_dir)
    train_cols, n_labels = _train_label_columns(cfg.raw_dir)

    head, tail = head_tail_split(train_cols, n_labels, cfg.head_frac)
    print(f"rótulos: {n_labels} | cabeça: {len(head)} | cauda: {len(tail)} "
          f"| agregação: {cfg.aggregation} | cutoff: {cfg.cutoff}")

    sr = build_index(ds.train.texts, cfg)
    runs = retrieve(sr, ds.test.texts, train_cols, head, tail, cfg)
    write_trec(runs, cfg.out_path, cfg.tag)
    print(f"run TREC salvo em {cfg.out_path} ({len(runs)} queries)")


if __name__ == "__main__":
    main()
