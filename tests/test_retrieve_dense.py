"""Testes das funções puras de src/retrieve_dense.py.

Importante: importar o módulo NÃO carrega torch/transformers/pytorch-metric-learning
(eles só são importados dentro de build_encoder/build_loss). Aqui não tocamos no
BERT real — usamos FakeEncoder (vetores determinísticos), então os testes são
determinísticos, instantâneos e leves. O teste da loss/miner é pulado se o
pytorch-metric-learning não estiver instalado (suíte mínima fica verde).
"""
import subprocess
import sys

import numpy as np
import pytest

from src.retrieve_dense import (
    DenseConfig,
    FakeEncoder,
    build_relevance_map,
    build_train_pairs,
    rank_per_class,
    rank_per_class_chunked,
    resolve_label_texts,
)
from src.retrieve_sparse import write_trec
from src.splits import Fold, PooledData


def test_import_nao_puxa_stack_pesada():
    # contrato de import lazy: importar o módulo NÃO pode carregar transformers nem
    # pytorch-metric-learning. Verificado num interpretador LIMPO (subprocess), pois
    # outro arquivo de teste pode já ter importado essas libs no sys.modules da suíte.
    code = (
        "import sys; import src.retrieve_dense as rd; "
        "assert 'transformers' not in sys.modules, 'transformers importado no load'; "
        "assert 'pytorch_metric_learning' not in sys.modules, 'pml importado no load'; "
        "assert not rd._PML_CACHE, '_pml_classes() chamado no load'; "
        "print('ok')"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert "ok" in res.stdout


# ─────────────────────────── resolve_label_texts (o seam opcional) ─────────────────

def test_resolve_label_texts_llm_concatena_nome_e_descricao():
    vocab = ["fisheries", "2164", "customs duties"]
    desc = {0: "rules about fishing and the sea", 2: "taxes on imported goods"}
    out = resolve_label_texts(vocab, desc, "LLM")
    assert out[0] == "fisheries rules about fishing and the sea"
    assert out[1] == "2164"                       # sem descrição → fallback ao nome cru
    assert out[2] == "customs duties taxes on imported goods"


def test_resolve_label_texts_none_usa_so_o_nome():
    vocab = ["fisheries", "customs duties"]
    out = resolve_label_texts(vocab, {0: "ignorada"}, "NONE")
    assert out == ["fisheries", "customs duties"]  # descrição ignorada em NONE


def test_resolve_label_texts_dict_vazio_e_fallback_total():
    # arquivo de RAG-labels ausente (dict vazio) → todos caem no nome cru
    vocab = ["a", "b", "c"]
    assert resolve_label_texts(vocab, {}, "LLM") == ["a", "b", "c"]


def test_resolve_label_texts_rejeita_enhancement_invalido():
    with pytest.raises(ValueError, match="LLM.*NONE|label_enhancement"):
        resolve_label_texts(["a"], {}, "PMI")


# ─────────────────────────── relevance map / pares de treino ───────────────────────

def test_build_relevance_map():
    rmap = build_relevance_map([[0, 1], [2], []])
    assert rmap == {0: {0, 1}, 1: {2}, 2: set()}


def test_build_train_pairs_explode_so_o_corpus_do_fold():
    pooled = PooledData(
        texts=["d0", "d1", "d2", "d3"],
        label_cols=[[10], [20, 21], [30], [40]],
        n_labels=50,
    )
    fold = Fold(fold_id=0, train_idx=np.array([0, 1, 3]), test_idx=np.array([2]))
    pairs = build_train_pairs(fold, pooled)
    # doc2 é query (fora do treino) → não aparece; doc1 gera 2 pares
    assert pairs == [(0, 10), (1, 20), (1, 21), (3, 40)]


# ─────────────────────────── rank_per_class (cosine + split cabeça/cauda) ──────────

def test_rank_per_class_separa_cabeca_e_cauda_e_trunca():
    # 1 query, 4 rótulos; cabeça={0,1}, cauda={2,3}; num_labels=1 por classe
    # vetores escolhidos para uma ordem de similaridade conhecida com a query
    q = np.array([[1.0, 0.0]], dtype=np.float32)
    labels = np.array([
        [1.0, 0.0],    # col0 (cabeça) sim=1.0
        [0.6, 0.8],    # col1 (cabeça) sim=0.6
        [0.0, 1.0],    # col2 (cauda) sim=0.0
        [0.8, 0.6],    # col3 (cauda) sim=0.8
    ], dtype=np.float32)
    runs = rank_per_class(q, ["7"], labels, [0, 1, 2, 3], head={0, 1}, tail={2, 3}, num_labels=1)
    # top-1 cabeça = col0 (1.0); top-1 cauda = col3 (0.8)
    cols = {c for c, _ in runs["7"]}
    assert cols == {0, 3}
    by_col = dict(runs["7"])
    assert by_col[0] == pytest.approx(1.0, abs=1e-5)
    assert by_col[3] == pytest.approx(0.8, abs=1e-5)


def test_rank_per_class_mantem_64_mais_64_por_padrao():
    rng = np.random.RandomState(0)
    n_labels = 300
    q = rng.randn(5, 8).astype(np.float32)
    labels = rng.randn(n_labels, 8).astype(np.float32)
    head = set(range(50))            # 50 de cabeça
    tail = set(range(50, n_labels))  # 250 de cauda
    runs = rank_per_class(q, [str(i) for i in range(5)], labels, list(range(n_labels)),
                          head, tail, num_labels=64)
    for items in runs.values():
        cabeca = [c for c, _ in items if c in head]
        cauda = [c for c, _ in items if c in tail]
        assert len(cabeca) == 50    # só há 50 de cabeça (min(64, 50))
        assert len(cauda) == 64     # trunca a cauda em 64
        assert len(items) == 114


def test_rank_per_class_classe_vazia_nao_quebra():
    q = np.array([[1.0, 0.0]], dtype=np.float32)
    labels = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    runs = rank_per_class(q, ["0"], labels, [0, 1], head={0, 1}, tail=set(), num_labels=5)
    assert {c for c, _ in runs["0"]} == {0, 1}   # cauda vazia: só cabeça


# ─────────────── inferência chunkada (Amazon-670K) — equivalência ao numpy ──────────

def _runs_equivalentes(a: dict, b: dict, tol: float = 1e-4) -> None:
    """Mesma seleção de rótulos por query e scores próximos (fp16/ordem do topk)."""
    assert a.keys() == b.keys()
    for qid in a:
        da, db = dict(a[qid]), dict(b[qid])
        assert da.keys() == db.keys(), f"qid {qid}: rótulos divergem"
        for col in da:
            assert da[col] == pytest.approx(db[col], abs=tol)


def test_rank_per_class_chunked_equivale_ao_numpy():
    # vocabulário "grande" sintético; vários blocos de query (chunk_size < Nq)
    rng = np.random.RandomState(7)
    n_labels, n_q, dim = 400, 37, 8
    q = rng.randn(n_q, dim).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)          # L2 (como o encoder)
    labels = rng.randn(n_labels, dim).astype(np.float32)
    labels /= np.linalg.norm(labels, axis=1, keepdims=True)
    head = set(range(80))
    tail = set(range(80, n_labels))
    qids = [str(i) for i in range(n_q)]

    exato = rank_per_class(q, qids, labels, list(range(n_labels)), head, tail, num_labels=64)
    chunk = rank_per_class_chunked(q, qids, labels, list(range(n_labels)), head, tail,
                                   num_labels=64, chunk_size=10, device="cpu")
    _runs_equivalentes(exato, chunk)


def test_rank_per_class_chunked_classe_vazia_nao_quebra():
    q = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    labels = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    chunk = rank_per_class_chunked(q, ["0", "1"], labels, [0, 1],
                                   head={0, 1}, tail=set(), num_labels=5,
                                   chunk_size=1, device="cpu")
    assert {c for c, _ in chunk["0"]} == {0, 1}   # cauda vazia: só cabeça, sem erro


# ─────────────────────── FakeEncoder ponta-a-ponta → run TREC ──────────────────────

def test_fake_encoder_deterministico_e_normalizado():
    fe = FakeEncoder(dim=16)
    a = fe.encode(["foo", "bar"])
    b = fe.encode(["foo", "bar"])
    assert np.allclose(a, b)                                  # determinístico
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5)  # L2-normalizado


def test_fluxo_inferencia_fake_encoder_escreve_trec(tmp_path):
    fe = FakeEncoder(dim=32)
    label_texts = [f"label {i}" for i in range(20)]
    query_texts = ["doc a", "doc b", "doc c"]
    query_ids = ["100", "101", "102"]
    head = set(range(5))
    tail = set(range(5, 20))

    text_rpr = fe.encode(query_texts)
    label_rpr = fe.encode(label_texts)
    runs = rank_per_class(text_rpr, query_ids, label_rpr, list(range(20)),
                          head, tail, num_labels=3)

    out = tmp_path / "dense.fold0.trec"
    write_trec(runs, str(out), tag="dense")
    lines = out.read_text(encoding="utf-8").splitlines()
    # 3 queries × (3 cabeça + 3 cauda) = 18 linhas
    assert len(lines) == 18
    # formato e ordenação por score desc dentro de cada query
    assert all(parts[1] == "Q0" and parts[5] == "dense"
               for parts in (l.split() for l in lines))
    first = lines[0].split()
    assert first[0] == "100" and first[2].startswith("label_") and first[3] == "1"


def test_dense_config_paths():
    cfg = DenseConfig()
    assert cfg.fold_out_path(2).endswith("runs/dense.fold2.trec")
    assert cfg.fold_rag_path(3).endswith("rag-labels/fold3/labels_descriptions.jsonl")


# ─────────────────────── loss + miner (pulado sem pytorch-metric-learning) ─────────

def test_relevance_miner_marca_pares_corretos():
    pytest.importorskip("pytorch_metric_learning")
    import torch

    from src.retrieve_dense import build_miner

    # batch: textos globais [0, 1]; rótulos [10, 11]; gold: doc0->{10}, doc1->{11}
    miner = build_miner({0: {10}, 1: {11}})
    text_ids = torch.tensor([0, 1])
    label_ids = torch.tensor([10, 11])
    a1, p, a2, n = miner.mine(text_ids, label_ids)
    pos = set(zip(a1.tolist(), p.tolist()))
    neg = set(zip(a2.tolist(), n.tolist()))
    assert pos == {(0, 0), (1, 1)}    # (text_i, label_j) positivos
    assert neg == {(0, 1), (1, 0)}    # cruzados são negativos


def test_loss_roda_e_retorna_escalar_positivo():
    pytest.importorskip("pytorch_metric_learning")
    import torch

    from src.retrieve_dense import build_loss

    loss_fn = build_loss({0: {10}, 1: {11}}, temperature=0.07)
    torch.manual_seed(0)
    text_rpr = torch.nn.functional.normalize(torch.randn(2, 8), dim=1)
    label_rpr = torch.nn.functional.normalize(torch.randn(2, 8), dim=1)
    loss = loss_fn(torch.tensor([0, 1]), text_rpr, torch.tensor([10, 11]), label_rpr)
    assert loss.item() > 0.0
