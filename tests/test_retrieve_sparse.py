"""Testes das funções puras de src/retrieve_sparse.py.

Importante: importar o módulo NÃO carrega o `retriv` (ele só é importado dentro
de build_index). Aqui não tocamos no BM25 real — usamos um FakeSR que devolve
vizinhos fixos, então os testes são determinísticos, instantâneos e leves.
"""
import pytest

from src.retrieve_sparse import (
    SparseConfig,
    head_tail_split,
    retrieve,
    write_trec,
)


class FakeSR:
    """Dublê de SparseRetriever: bsearch devolve um mapa fixo {qid: {docid: score}}."""

    def __init__(self, results):
        self._results = results

    def bsearch(self, queries, cutoff):
        return self._results


def test_head_tail_split_partitions_by_frequency():
    # freq: label0=3, label1=2, label2=1, label3=0
    train_cols = [[0, 1], [0, 1], [0], [2]]
    head, tail = head_tail_split(train_cols, n_labels=4, head_frac=0.25)
    assert head == {0}                 # round(0.25*4)=1 rótulo mais frequente
    assert tail == {1, 2, 3}
    assert head.isdisjoint(tail)
    assert head | tail == {0, 1, 2, 3}


def test_head_tail_split_all_tail_when_head_frac_zero():
    head, tail = head_tail_split([[0], [1]], n_labels=2, head_frac=0.0)
    assert head == set()
    assert tail == {0, 1}


def test_retrieve_sum_aggregation_and_bucketing():
    cfg = SparseConfig(cutoff=10, num_labels=2, aggregation="sum")
    train_cols = [[0, 1], [1, 2], [3]]            # rótulos-gold de cada doc de treino
    head, tail = {0, 1}, {2, 3}
    # query "0" recuperou doc0 (2.0), doc1 (1.0), doc2 (0.5)
    sr = FakeSR({"0": {"0": 2.0, "1": 1.0, "2": 0.5}})

    runs = retrieve(sr, ["q"], train_cols, head, tail, cfg)

    # head: label0=2.0, label1=2.0+1.0=3.0 ; tail: label2=1.0, label3=0.5
    assert runs["0"] == [(1, 3.0), (0, 2.0), (2, 1.0), (3, 0.5)]


def test_retrieve_max_aggregation():
    cfg = SparseConfig(cutoff=10, num_labels=5, aggregation="max")
    train_cols = [[0], [0]]                        # ambos os docs têm o rótulo 0
    head, tail = {0}, set()
    sr = FakeSR({"0": {"0": 2.0, "1": 3.0}})       # doc1 tem score maior

    runs = retrieve(sr, ["q"], train_cols, head, tail, cfg)

    assert runs["0"] == [(0, 3.0)]                  # max(2.0, 3.0), não a soma (5.0)


def test_retrieve_truncates_to_num_labels_per_bucket():
    cfg = SparseConfig(cutoff=10, num_labels=1, aggregation="sum")
    train_cols = [[0], [1]]                         # dois rótulos de cabeça concorrendo
    head, tail = {0, 1}, set()
    sr = FakeSR({"0": {"0": 1.0, "1": 5.0}})

    runs = retrieve(sr, ["q"], train_cols, head, tail, cfg)

    assert runs["0"] == [(1, 5.0)]                  # só o top-1 da cabeça sobrevive


def test_retrieve_rejects_invalid_aggregation():
    cfg = SparseConfig(aggregation="média")
    sr = FakeSR({"0": {"0": 1.0}})
    with pytest.raises(ValueError, match="sum.*max|aggregation"):
        retrieve(sr, ["q"], [[0]], {0}, set(), cfg)


def test_write_trec_format_and_ordering(tmp_path):
    out = tmp_path / "run.trec"
    # itens fora de ordem de propósito: write_trec deve ordenar por score desc
    runs = {"0": [(0, 2.0), (1, 3.0)], "1": [(5, 0.5)]}

    write_trec(runs, str(out), tag="bm25")

    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines == [
        "0 Q0 label_1 1 3.000000 bm25",
        "0 Q0 label_0 2 2.000000 bm25",
        "1 Q0 label_5 1 0.500000 bm25",
    ]


def test_write_trec_creates_parent_dir(tmp_path):
    out = tmp_path / "subdir" / "nested" / "run.trec"
    write_trec({"0": [(0, 1.0)]}, str(out), tag="x")
    assert out.exists()
    assert out.read_text(encoding="utf-8").strip() == "0 Q0 label_0 1 1.000000 x"
