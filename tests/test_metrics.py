"""Testes de src/metrics.py.

Duas camadas:
  - Funções puras (parsing, segmentação cabeça/cauda, agregação): determinísticas,
    sem ranx.
  - Métricas via ranx (fonte da verdade): conferidas num exemplo calculável na mão.
    Pulado se o ranx não estiver instalado.

Importar src.metrics NÃO carrega o ranx (import lazy dentro de evaluate).
"""
import subprocess
import sys

import pytest

from src.metrics import (
    MetricsConfig,
    _runs_to_sparse,
    aggregate,
    build_qrels,
    evaluate,
    evaluate_segmented,
    label_to_col,
    metric_names,
    segment,
)
from src.splits import PooledData


def test_import_nao_puxa_ranx():
    code = (
        "import sys, src.metrics; "
        "assert 'ranx' not in sys.modules, 'ranx importado no load de src.metrics'; "
        "print('ok')"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert "ok" in res.stdout


def test_label_to_col():
    assert label_to_col("label_0") == 0
    assert label_to_col("label_123") == 123


def test_metric_names():
    names = metric_names((1, 5), ("precision", "ndcg"))
    assert names == ["precision@1", "precision@5", "ndcg@1", "ndcg@5"]


def test_build_qrels():
    pooled = PooledData(texts=["a", "b"], label_cols=[[0, 2], [5]], n_labels=10)
    q = build_qrels(pooled)
    assert q == {"0": {"label_0": 1.0, "label_2": 1.0}, "1": {"label_5": 1.0}}


# ─────────────────────────── segmentação cabeça/cauda ──────────────────────────────

def test_segment_restringe_run_e_gold():
    run = {"q": {"label_0": 5.0, "label_1": 4.0, "label_2": 3.0}}
    qrels = {"q": {"label_0": 1.0, "label_2": 1.0}}
    run_h, qrels_h = segment(run, qrels, cols={0, 1})       # cabeça = {0,1}
    assert qrels_h == {"q": {"label_0": 1.0}}                # só o gold de cabeça
    assert run_h == {"q": {"label_0": 5.0, "label_1": 4.0}}  # só candidatos de cabeça
    run_t, qrels_t = segment(run, qrels, cols={2})          # cauda = {2}
    assert qrels_t == {"q": {"label_2": 1.0}}
    assert run_t == {"q": {"label_2": 3.0}}


def test_segment_descarta_query_sem_gold_no_segmento():
    run = {"q": {"label_0": 1.0}, "q2": {"label_9": 1.0}}
    qrels = {"q": {"label_0": 1.0}, "q2": {"label_9": 1.0}}
    # cauda = {9}: a query "q" (gold só na cabeça) não é avaliável na cauda
    run_t, qrels_t = segment(run, qrels, cols={9})
    assert set(qrels_t) == {"q2"}
    assert "q" not in run_t


# ─────────────────────────── agregação entre folds ─────────────────────────────────

def test_aggregate_media_e_desvio():
    per_fold = [{"precision@1": 1.0}, {"precision@1": 0.0}]
    agg = aggregate(per_fold)
    mean, std = agg["precision@1"]
    assert mean == pytest.approx(0.5)
    assert std == pytest.approx(0.5)   # desvio populacional (ddof=0)


# ─────────────────────── métricas via ranx (calculáveis na mão) ────────────────────

def test_runs_to_sparse_ordem_e_alinhamento():
    # parte PURA do PSP (sem xclib): pred guarda a ORDEM como escore positivo decrescente;
    # true binário; mesma ordem de linhas; só queries com gold.
    run = {"5": {"label_0": 0.1, "label_2": 0.9}, "7": {"label_1": 0.5}}
    qrels = {"5": {"label_0": 1.0, "label_2": 1.0}, "7": {"label_1": 1.0}}
    pred, true = _runs_to_sparse(run, qrels, 3)
    assert pred.shape == (2, 3) and true.shape == (2, 3)
    assert true.nnz == 3                                   # 3 rótulos gold no total
    row5 = pred[0].toarray()[0]                            # qids ordenados → "5" é a linha 0
    assert row5[2] > row5[0] > 0                           # label_2 (0.9) ranqueado acima de label_0 (0.1)


def test_evaluate_psp_exige_inv_psp():
    # 'psp'/'psndcg' sem inv_psp → erro claro (não importa o xclib silenciosamente)
    run = {"q": {"label_0": 1.0}}
    qrels = {"q": {"label_0": 1.0}}
    with pytest.raises(ValueError):
        evaluate(run, qrels, (1,), ("psp",))
    # caminho ranx puro segue funcionando sem inv_psp
    assert "precision@1" in evaluate(run, qrels, (1,), ("precision",))


def test_evaluate_valores_calculaveis():
    pytest.importorskip("ranx")
    from src.metrics import evaluate

    # gold = {1, 3}; ranking por score: label_1(3) > label_2(2) > label_3(1)
    run = {"q": {"label_1": 3.0, "label_2": 2.0, "label_3": 1.0}}
    qrels = {"q": {"label_1": 1.0, "label_3": 1.0}}
    res = evaluate(run, qrels, ks=(1, 2, 3), kinds=("precision", "recall"))
    assert res["precision@1"] == pytest.approx(1.0)     # top1 = label_1 (gold)
    assert res["precision@2"] == pytest.approx(0.5)     # top2: 1 gold de 2
    assert res["precision@3"] == pytest.approx(2 / 3)   # top3: 2 gold de 3
    assert res["recall@3"] == pytest.approx(1.0)        # 2 gold recuperados de 2


def test_evaluate_segmented_separa_cabeca_e_cauda():
    pytest.importorskip("ranx")
    cfg = MetricsConfig(ks=(1,), kinds=("precision",))
    # cabeça={0,1}, cauda={2,3}
    run = {"q": {"label_0": 9.0, "label_1": 1.0, "label_2": 8.0, "label_3": 1.0}}
    qrels = {"q": {"label_0": 1.0, "label_2": 1.0}}      # 1 gold cabeça, 1 gold cauda
    seg = evaluate_segmented(run, qrels, head={0, 1}, tail={2, 3}, cfg=cfg)
    # cabeça: top1 entre {0,1} = label_0 (gold) → P@1=1 ; cauda: top1 entre {2,3} = label_2 (gold) → P@1=1
    assert seg["head"]["precision@1"] == pytest.approx(1.0)
    assert seg["tail"]["precision@1"] == pytest.approx(1.0)
    assert seg["overall"]["precision@1"] == pytest.approx(1.0)  # top1 global = label_0 (gold)


def test_evaluate_query_sem_candidato_no_segmento_da_zero():
    pytest.importorskip("ranx")
    from src.metrics import evaluate

    # gold de cauda existe, mas o run não tem candidato de cauda → P@1 = 0 (não quebra)
    run = {"q": {}}
    qrels = {"q": {"label_2": 1.0}}
    res = evaluate(run, qrels, ks=(1,), kinds=("precision",))
    assert res["precision@1"] == pytest.approx(0.0)
