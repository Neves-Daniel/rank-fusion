"""Recuperador esparso (BM25) para XMTC — metodologia xCoRetriev / RAG-Fuse.

Abordagem: kNN léxico ("documentos parecidos têm rótulos parecidos").
  1. Indexa os documentos de TREINO com BM25.
  2. Para cada documento de TESTE (query), recupera os top-`cutoff` vizinhos
     de treino, cada um com sua nota BM25.
  3. Cada vizinho "vota" nos seus próprios rótulos-gold, com peso = nota BM25.
     O score de um rótulo é o MÁX (default, fiel ao paper) das notas dos vizinhos
     que o têm — xCoRetriev/RAG-Fuse: "pontuação igual ao maior valor de relevância
     entre t e cada texto recuperado". SOMA (CombSUM) fica como variante experimental.
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
    python -m src.retrieve_sparse           # roda no Eurlex-4K, gera runs/sparse.trec
    python src/retrieve_sparse.py           # idem, a partir da raiz do projeto
"""
from __future__ import annotations

import heapq
import os
import sys
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

if __package__ in {None, ""}:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_dataset
from src.splits import load_pooled, make_folds


@dataclass
class SparseConfig:
    raw_dir: str = "data/eurlex4k/raw"
    out_path: str = "data/eurlex4k/runs/sparse.trec"   # usado pelo modo single-split (legado/demo)
    runs_dir: str = "data/eurlex4k/runs"               # modo CV: 1 run por fold aqui
    out_template: str = "sparse.fold{fold}.trec"       # nome do run de cada fold
    retriv_base_path: str = "data/.retriv"
    cutoff: int = 100          # nº de vizinhos (docs de treino) recuperados por query
    num_labels: int = 64       # nº de rótulos mantidos POR classe (cabeça/cauda)
    query_batch_size: int = 128  # queries por lote no bsearch (limita a memória; ver retrieve)
    dedup_query_terms: bool = True  # deduplica tokens da query; ver retrieve (evita segfault do retriv)
    aggregation: str = "max"   # paper (xCoRetriev/RAG-Fuse): "maior valor de relevância" = CombMAX. "sum" (CombSUM) é variante.
    head_frac: float = 0.20    # 20% rótulos mais frequentes = cabeça (Pareto)
    n_folds: int = 5           # k da validação cruzada (artigo: 5-fold CV)
    seed: int = 42             # semente da atribuição de folds (reprodutibilidade)
    resume: bool = True        # pula fold cujo .trec já existe (resume-safe; ver run_cv)
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    tag: str = "bm25"

    def fold_out_path(self, fold_id: int) -> str:
        return os.path.join(self.runs_dir, self.out_template.format(fold=fold_id))


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
    from retriv import SparseRetriever, set_base_path

    set_base_path(cfg.retriv_base_path)

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
    query_ids: list[str] | None = None,
) -> dict[str, list[tuple[int, float]]]:
    """Recupera vizinhos e agrega seus rótulos em scores, por classe (cabeça/cauda).

    Retorna {qid: [(coluna_rótulo, score), ...]} com até num_labels*2 candidatos.

    `query_ids` é o rótulo de cada query no run (qid). No modo CV é o índice GLOBAL
    do doc agrupado, para o gold ser recuperável depois por esse mesmo índice. Se
    omitido, usa a posição local `str(i)` (compatível com o modo single-split/demo).

    Dedup de query (evita segfault do retriv): documentos longos usados como
    query estouram a pilha do retriv. O retriv NÃO deduplica os termos da query
    — monta um array de postings por token — e `union_sorted_multi` (numba) é
    recursiva com profundidade ≈ nº de termos / 2. Um doc do Eurlex com ~28 mil
    tokens gera recursão de ~14 mil níveis → estouro da pilha nativa → segfault.
    Pior: o kernel roda numa worker thread do numba, cuja pilha é pequena (teto
    medido ~5 mil termos), e isso não se contorna por env var nem batch size.

    Por isso deduplicamos os tokens (preservando a ordem) antes de buscar: o
    comprimento da lista cai para o nº de termos DISTINTOS (q31: 28.301 → 1.842,
    bem abaixo do teto). Preserva todo o vocabulário do documento; só remove o
    peso por repetição do termo na query (query-TF) — escolha de pesos binários,
    defensável para este kNN de rótulos.
    """
    batch = max(1, cfg.query_batch_size)
    max_terms = 0
    runs: dict[str, list[tuple[int, float]]] = {}
    if query_ids is None:
        query_ids = [str(i) for i in range(len(test_texts))]

    def _query_text(i: int) -> str:
        nonlocal max_terms
        toks = test_texts[i].split()
        if cfg.dedup_query_terms:
            seen: set[str] = set()
            toks = [t for t in toks if not (t in seen or seen.add(t))]
        max_terms = max(max_terms, len(toks))
        return " ".join(toks)

    for start in range(0, len(test_texts), batch):
        queries = [
            {"id": query_ids[i], "text": _query_text(i)}
            for i in range(start, min(start + batch, len(test_texts)))
        ]
        results = sr.bsearch(queries=queries, cutoff=cfg.cutoff)  # {qid: {docid: score}}

        for qid, neighbors in results.items():
            runs[qid] = _aggregate_neighbors(neighbors, train_label_cols, head, tail, cfg)

    if cfg.dedup_query_terms:
        print(f"dedup de termos na query: maior query = {max_terms} termos distintos")
    return runs


def _aggregate_neighbors(
    neighbors: dict[str, float],
    train_label_cols: list[list[int]],
    head: set[int],
    tail: set[int],
    cfg: SparseConfig,
) -> list[tuple[int, float]]:
    """Agrega os rótulos dos vizinhos recuperados para uma query."""
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
    return top_head + top_tail


def write_trec(runs: dict[str, list[tuple[int, float]]], out_path: str, tag: str) -> None:
    """Escreve o run no formato TREC: `qid Q0 label_id rank score tag`."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for qid, items in runs.items():
            ranked = sorted(items, key=lambda kv: kv[1], reverse=True)
            for rank, (col, score) in enumerate(ranked, start=1):
                fh.write(f"{qid} Q0 label_{col} {rank} {score:.6f} {tag}\n")


def run_single_split(cfg: SparseConfig | None = None) -> None:
    """Modo legado: split fixo treino/teste do PECOS → 1 run em cfg.out_path.

    Mantido para inspeção rápida; o protocolo oficial do projeto é 5-fold CV
    (run_cv), fiel ao artigo. Ver src/splits.py.
    """
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


def run_cv(cfg: SparseConfig | None = None, only_fold: int | None = None) -> None:
    """Protocolo oficial: 5-fold CV sobre o dataset agrupado (treino+teste).

    Para cada fold, indexa o corpus (4/5 dos docs) com BM25, usa o 1/5 restante
    como queries e escreve um run TREC por fold (qid = índice GLOBAL do doc, ver
    src/splits.py). A partição cabeça/cauda é GLOBAL (frequências sobre os N docs
    agrupados), fiel à definição do artigo.

    Resume-safe e isolável por fold (`only_fold`): com corpora grandes (ex.: Wiki10,
    texto integral), rodar os 5 folds num único processo acumula memória entre folds
    (objetos do retriv, page cache contra o limite do cgroup) e pode estourar o OOM já
    na indexação de um fold tardio. Rodar 1 fold por processo (`--fold N`) garante que
    o SO recupera tudo ao sair; `resume` pula folds cujo `.trec` já existe. Mesmo no
    modo todos-os-folds, liberamos o índice e damos gc.collect() a cada iteração.
    """
    import gc

    cfg = cfg or SparseConfig()
    pooled = load_pooled(cfg.raw_dir)

    head, tail = head_tail_split(pooled.label_cols, pooled.n_labels, cfg.head_frac)
    print(f"agrupado: {len(pooled)} docs | rótulos: {pooled.n_labels} "
          f"| cabeça: {len(head)} | cauda: {len(tail)} (global, Pareto) "
          f"| agregação: {cfg.aggregation} | cutoff: {cfg.cutoff} | {cfg.n_folds}-fold (seed={cfg.seed})")

    folds = make_folds(len(pooled), k=cfg.n_folds, seed=cfg.seed)
    for fold in folds:
        if only_fold is not None and fold.fold_id != only_fold:
            continue

        out_path = cfg.fold_out_path(fold.fold_id)
        if cfg.resume and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            print(f"\n── fold {fold.fold_id}: já existe {out_path} — pulando (resume) ──")
            continue

        corpus_texts = [pooled.texts[i] for i in fold.train_idx]
        corpus_cols = [pooled.label_cols[i] for i in fold.train_idx]
        query_texts = [pooled.texts[i] for i in fold.test_idx]
        query_ids = [str(int(i)) for i in fold.test_idx]   # índice GLOBAL agrupado

        print(f"\n── fold {fold.fold_id}: corpus {len(corpus_texts)} | queries {len(query_texts)} ──")
        sr = build_index(corpus_texts, cfg)
        runs = retrieve(sr, query_texts, corpus_cols, head, tail, cfg, query_ids=query_ids)

        write_trec(runs, out_path, cfg.tag)
        print(f"run TREC salvo em {out_path} ({len(runs)} queries)")

        del sr, runs, corpus_texts, corpus_cols, query_texts   # libera entre folds
        gc.collect()


def main(cfg: SparseConfig | None = None) -> None:
    import argparse

    from src.data import add_dataset_arg, apply_dataset

    parser = argparse.ArgumentParser(description="Recuperador esparso (BM25 kNN, 5-fold CV)")
    add_dataset_arg(parser)
    parser.add_argument("--fold", type=int, default=None,
                        help="roda só este fold (isolamento de memória / paralelizar)")
    parser.add_argument("--query-batch-size", type=int, default=None,
                        help="queries por lote no bsearch; menor = menos pico de memória "
                             "(ex.: 16 em datasets grandes tipo AmazonCat)")
    parser.add_argument("--no-resume", action="store_true",
                        help="refaz o fold mesmo que o .trec já exista")
    args, _ = parser.parse_known_args()

    cfg = cfg or SparseConfig()
    apply_dataset(cfg, args.dataset)
    if args.query_batch_size:
        cfg.query_batch_size = args.query_batch_size
    if args.no_resume:
        cfg.resume = False
    run_cv(cfg, only_fold=args.fold)


if __name__ == "__main__":
    main()
