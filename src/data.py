"""Carregamento dos datasets XMTC no formato PECOS / X-Transformer.

Espelho usado: HuggingFace thekop79/EURLex-4K (ver scripts/download_eurlex.sh).
Layout esperado em <raw_dir>:
  trn_X.txt / tst_X.txt : um documento por linha (texto pré-processado)
  Y.trn.npz / Y.tst.npz : matriz esparsa CSR (n_docs x n_labels), valores binários
  Y.txt                 : vocabulário de rótulos, um por linha (linha i = coluna i de Y)

Os rótulos de cada documento são as colunas não-nulas da sua linha em Y, mapeadas
para a string correspondente em Y.txt. Usamos sempre o ID textual do rótulo
(não o índice de coluna) para que os rankings sejam comparáveis entre datasets.

Uso:
    from src.data import load_dataset, dataset_stats
    ds = load_dataset("data/eurlex4k/raw")
    print(dataset_stats(ds))
    ds.train.texts[0], ds.train.labels[0]   # texto e lista de rótulos do doc 0
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import scipy.sparse as sp


@dataclass
class Split:
    texts: list[str]
    labels: list[list[str]]          # IDs textuais de rótulo por documento
    name: str = ""

    def __len__(self) -> int:
        return len(self.texts)


@dataclass
class Dataset:
    train: Split
    test: Split
    label_vocab: list[str]           # linha i = rótulo da coluna i em Y


def dataset_paths(name: str) -> dict[str, str]:
    """Caminhos canônicos de um dataset em ``data/<name>/`` (multi-dataset).

    Mantém a convenção usada em todo o pipeline: ``raw/`` (insumos baixados),
    ``runs/`` (runs TREC), ``rag-labels/`` (descrições por fold) e ``results/``.
    """
    base = f"data/{name}"
    return {
        "raw_dir": f"{base}/raw",
        "runs_dir": f"{base}/runs",
        "rag_labels_dir": f"{base}/rag-labels",
        "results_dir": f"{base}/results",
    }


def apply_dataset(cfg, name: str) -> None:
    """Aponta os caminhos de uma config (dataclass) para ``data/<name>/...``.

    Sobrescreve só os atributos que a config tiver — cada módulo nomeia seus
    paths de um jeito (``out_dir`` no label_desc, ``out_csv`` no gridsearch etc.),
    então cobrimos os nomes conhecidos sem acoplar a uma config específica.
    Atenção: só toca atributos que JÁ são string — alguns nomes (ex.: ``out_path``
    na FusionConfig) são MÉTODOS, e não podem ser clobberados com um caminho.
    """
    p = dataset_paths(name)
    overrides = {
        "raw_dir": p["raw_dir"],
        "runs_dir": p["runs_dir"],
        "rag_labels_dir": p["rag_labels_dir"],
        "out_dir": p["rag_labels_dir"],                # label_desc: rag-labels
        "out_path": f"{p['runs_dir']}/sparse.trec",    # sparse: modo single-split (legado)
        "out_csv": f"{p['results_dir']}/gridsearch.csv",  # gridsearch
    }
    for attr, value in overrides.items():
        if isinstance(getattr(cfg, attr, None), str):  # ignora métodos/ausentes
            setattr(cfg, attr, value)


def add_dataset_arg(parser, default: str = "eurlex4k") -> None:
    """Adiciona ``--dataset`` ao argparse de um CLI (default mantém retrocompat)."""
    parser.add_argument(
        "--dataset", type=str, default=default,
        help="nome do dataset em data/<nome>/ (default: eurlex4k)",
    )


def add_folds_arg(parser) -> None:
    """Adiciona ``--folds`` (subconjunto de folds a avaliar/fundir, ex.: ``0,1,2``).

    NÃO muda a partição (que é k=n_folds, congelada nos próprios runs `.trec`): só
    seleciona QUAIS folds entram na média de CV. Útil para datasets grandes onde se
    roda 3 dos 5 folds por orçamento de compute (mantendo treino 4/5, fiel ao artigo).
    """
    parser.add_argument(
        "--folds", type=str, default=None,
        help="subconjunto de folds, ex.: 0,1,2 (default: todos os n_folds). "
             "Não altera a partição k=n_folds — só quais folds entram na média.",
    )


def parse_folds(s: str | None) -> tuple[int, ...] | None:
    """``"0,1,2"`` → ``(0, 1, 2)``; ``None``/vazio → ``None`` (= todos os folds)."""
    if not s:
        return None
    return tuple(int(x) for x in s.split(",") if x.strip() != "")


def _read_lines(path: str) -> list[str]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Arquivo não encontrado: {path}\n"
            f"Rode antes o script de download do dataset (scripts/download_*.sh)"
        )
    with open(path, encoding="utf-8") as fh:
        return [line.rstrip("\n") for line in fh]


def _labels_from_matrix(Y: sp.spmatrix, vocab: list[str]) -> list[list[str]]:
    """Para cada linha (documento), lista os IDs de rótulo das colunas não-nulas."""
    Ycsr = Y.tocsr()
    if Ycsr.shape[1] != len(vocab):
        raise ValueError(
            f"Y tem {Ycsr.shape[1]} colunas mas Y.txt tem {len(vocab)} rótulos."
        )
    out: list[list[str]] = []
    indptr, indices = Ycsr.indptr, Ycsr.indices
    for i in range(Ycsr.shape[0]):
        cols = indices[indptr[i]:indptr[i + 1]]
        out.append([vocab[c] for c in cols])
    return out


def load_dataset(raw_dir: str) -> Dataset:
    vocab = _read_lines(os.path.join(raw_dir, "Y.txt"))

    trn_texts = _read_lines(os.path.join(raw_dir, "trn_X.txt"))
    tst_texts = _read_lines(os.path.join(raw_dir, "tst_X.txt"))

    Ytrn = sp.load_npz(os.path.join(raw_dir, "Y.trn.npz"))
    Ytst = sp.load_npz(os.path.join(raw_dir, "Y.tst.npz"))

    if Ytrn.shape[0] != len(trn_texts):
        raise ValueError(f"treino: {len(trn_texts)} textos vs {Ytrn.shape[0]} linhas em Y.trn")
    if Ytst.shape[0] != len(tst_texts):
        raise ValueError(f"teste: {len(tst_texts)} textos vs {Ytst.shape[0]} linhas em Y.tst")

    train = Split(trn_texts, _labels_from_matrix(Ytrn, vocab), "train")
    test = Split(tst_texts, _labels_from_matrix(Ytst, vocab), "test")
    return Dataset(train=train, test=test, label_vocab=vocab)


def dataset_stats(ds: Dataset) -> dict:
    n_per_doc = [len(l) for l in ds.train.labels]
    # rótulos efetivamente usados (treino+teste)
    used = set()
    for sp_ in (ds.train, ds.test):
        for labs in sp_.labels:
            used.update(labs)
    return {
        "n_train": len(ds.train),
        "n_test": len(ds.test),
        "n_labels_vocab": len(ds.label_vocab),
        "n_labels_used": len(used),
        "avg_labels_per_doc": round(sum(n_per_doc) / max(len(n_per_doc), 1), 3),
        "max_labels_per_doc": max(n_per_doc, default=0),
    }


if __name__ == "__main__":
    import sys

    raw = sys.argv[1] if len(sys.argv) > 1 else "data/eurlex4k/raw"
    ds = load_dataset(raw)
    print(f"Diretório: {raw}")
    for k, v in dataset_stats(ds).items():
        print(f"  {k:22s}: {v}")
    print("\nExemplo (treino[0]):")
    print("  texto :", ds.train.texts[0][:140], "...")
    print("  labels:", ds.train.labels[0][:10])
