"""Testes de src/gridsearch.py.

- rank_records: pura, determinística (sem ranx).
- evaluate_combo: integração com ranx (runs sintéticos em memória) — pulado sem ranx.
- import não puxa ranx (lazy via fusion/metrics).
"""
import subprocess
import sys

import pytest

from src.gridsearch import (
    PAPER_METHODS,
    PAPER_NORMS,
    GridConfig,
    rank_records,
)
from src.metrics import MetricsConfig


def test_import_nao_puxa_ranx():
    code = (
        "import sys, src.gridsearch; "
        "assert 'ranx' not in sys.modules, 'ranx importado no load de src.gridsearch'; "
        "print('ok')"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert "ok" in res.stdout


def test_grid_config_defaults_98_combos():
    cfg = GridConfig()
    assert len(cfg.norms) == 7 and len(cfg.methods) == 14   # 98 combinações
    assert cfg.select_segment == "tail" and cfg.select_metric == "ndcg@5"


def test_paper_subset_6x10():
    assert len(PAPER_NORMS) == 6 and len(PAPER_METHODS) == 10
    assert "combmnz" in PAPER_METHODS and "zmuv" in PAPER_NORMS
    # as extras NÃO entram no modo --paper
    assert "rrf" not in PAPER_METHODS and "minmaxinv" not in PAPER_NORMS


def test_rank_records_ordena_por_segmento_metrica():
    records = [
        {"norm": "a", "method": "x", "agg": {"tail": {"ndcg@5": (0.30, 0.0)}}},
        {"norm": "b", "method": "y", "agg": {"tail": {"ndcg@5": (0.50, 0.0)}}},
        {"norm": "c", "method": "z", "agg": {"tail": {"ndcg@5": (0.40, 0.0)}}},
    ]
    ranked = rank_records(records, "tail", "ndcg@5")
    assert [r["method"] for r in ranked] == ["y", "z", "x"]   # 0.50 > 0.40 > 0.30


def test_rank_records_ascendente():
    records = [
        {"norm": "a", "method": "x", "agg": {"tail": {"p@1": (0.3, 0.0)}}},
        {"norm": "b", "method": "y", "agg": {"tail": {"p@1": (0.1, 0.0)}}},
    ]
    ranked = rank_records(records, "tail", "p@1", descending=False)
    assert ranked[0]["method"] == "y"


def test_evaluate_combo_estrutura_e_segmentos():
    pytest.importorskip("ranx")
    from ranx import Run

    from src.gridsearch import evaluate_combo

    # 1 fold: esparso e denso de brinquedo; cabeça={0,1}, cauda={2}
    sparse = Run({"q": {"label_0": 1.0, "label_2": 0.5}}, name="sparse")
    dense = Run({"q": {"label_1": 2.0, "label_2": 1.0}}, name="dense")
    qrels = {"q": {"label_0": 1.0, "label_2": 1.0}}
    fold_runs = [(sparse, dense, qrels)]
    mcfg = MetricsConfig(ks=(1,), kinds=("precision",))

    agg = evaluate_combo("zmuv", "combmnz", fold_runs, head={0, 1}, tail={2}, mcfg=mcfg)
    assert set(agg) == {"overall", "head", "tail"}
    # cada métrica é (média, desvio); 1 fold → desvio 0
    mean, std = agg["tail"]["precision@1"]
    assert 0.0 <= mean <= 1.0 and std == pytest.approx(0.0)
    # label_2 (cauda) é gold e foi recuperado pelos dois → tail P@1 deve ser 1.0
    assert mean == pytest.approx(1.0)


def test_evaluate_combo_roda_para_varias_fusoes():
    pytest.importorskip("ranx")
    from ranx import Run

    from src.gridsearch import evaluate_combo
    from src.fusion import method_params

    sparse = Run({"q": {"label_0": 3.0, "label_2": 1.0}}, name="sparse")
    dense = Run({"q": {"label_1": 2.0, "label_2": 1.5}}, name="dense")
    qrels = {"q": {"label_0": 1.0, "label_2": 1.0}}     # gold em cabeça (0) e cauda (2)
    fold_runs = [(sparse, dense, qrels)]
    mcfg = MetricsConfig(ks=(1,), kinds=("ndcg",))
    for method in ("combmnz", "rrf", "rbc", "gmnz"):       # inclui as parametrizadas
        agg = evaluate_combo("minmax", method, fold_runs, {0, 1}, {2}, mcfg,
                             params=method_params(method))
        assert "ndcg@1" in agg["overall"]
