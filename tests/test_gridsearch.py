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
    _append_checkpoint,
    checkpoint_path,
    load_checkpoint,
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


def test_grid_config_defaults_175_combos():
    cfg = GridConfig()
    assert len(cfg.norms) == 7 and len(cfg.methods) == 25   # 175 combinações (grade completa)
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


def test_checkpoint_roundtrip_e_so_celulas_completas(tmp_path):
    # nomes de métrica que a célula precisa ter pra ser "completa"
    names = ["precision@1", "ndcg@1"]               # 2 métricas × 3 segmentos = 6
    ckpt = str(tmp_path / "grid.ckpt.csv")

    def rec(method, norm, val):
        agg = {s: {n: (val, 0.0) for n in names} for s in ("overall", "head", "tail")}
        return {"method": method, "norm": norm, "agg": agg}

    # grava 2 células completas
    _append_checkpoint(ckpt, rec("combmnz", "zmuv", 0.5), names)
    _append_checkpoint(ckpt, rec("rrf", "sum", 0.3), names)
    loaded = load_checkpoint(ckpt, names)
    assert set(loaded) == {("combmnz", "zmuv"), ("rrf", "sum")}
    assert loaded[("combmnz", "zmuv")]["tail"]["ndcg@1"] == (0.5, 0.0)

    # uma célula PARCIAL (faltando métricas) NÃO conta como feita
    with open(ckpt, "a") as fh:
        fh.write("wsum,zmuv,tail,precision@1,0.9,0.0\n")     # só 1 linha de 6
    loaded2 = load_checkpoint(ckpt, names)
    assert ("wsum", "zmuv") not in loaded2                   # incompleta → refazer


def test_checkpoint_path_deriva_do_csv():
    assert checkpoint_path("data/x/results/gridsearch.csv").endswith("gridsearch.ckpt.csv")


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


def test_skip_methods_filtra_a_grade():
    # filtro do --skip-methods + construção de all_cells (gridsearch.py:367)
    cfg = GridConfig()
    n_all = len(cfg.methods)
    skip = {"condorcet", "wcondorcet", "probfuse", "segfuse", "slidefuse"}
    cfg.methods = tuple(m for m in cfg.methods if m not in skip)
    cells = [(m, n) for m in cfg.methods for n in cfg.norms]
    assert len(cfg.methods) == n_all - len(skip)        # os 5 saíram
    assert not any(m in skip for m, _ in cells)         # nenhuma célula dos caros
    assert ("combmnz", "zmuv") in cells                 # os baratos permanecem


def test_subsample_fold_deterministico_e_alinhado():
    pytest.importorskip("ranx")
    from ranx import Run

    from src.fusion import run_to_dict
    from src.gridsearch import subsample_fold

    qs = [str(i) for i in range(6)]
    sparse = Run({q: {"label_0": 1.0, "label_2": 0.5} for q in qs}, name="sparse")
    dense = Run({q: {"label_1": 2.0} for q in qs}, name="dense")
    qrels_all = {q: {"label_0": 1.0} for q in qs}

    sp, de, qr = subsample_fold(sparse, dense, qrels_all, n=3)
    sp_q = set(run_to_dict(sp))
    assert len(sp_q) == 3                              # encolheu p/ n
    assert sp_q == set(run_to_dict(de)) == set(qr)     # runs E qrels no MESMO conjunto
    assert sp_q <= set(qs)                             # subconjunto do fold
    # determinístico (seed fixa) entre chamadas
    sp2, _, _ = subsample_fold(sparse, dense, qrels_all, n=3)
    assert set(run_to_dict(sp2)) == sp_q
    # n=0 ou n≥|fold| → fold inteiro, sem subamostrar
    assert len(run_to_dict(subsample_fold(sparse, dense, qrels_all, n=0)[0])) == 6
    assert len(run_to_dict(subsample_fold(sparse, dense, qrels_all, n=99)[0])) == 6


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


def test_evaluate_combo_supervised_cv_aninhada():
    pytest.importorskip("ranx")
    from ranx import Run

    from src.gridsearch import evaluate_combo_supervised

    # 2 folds (qids disjuntos): fold k treina no outro fold. cauda={2}
    f0 = (
        Run({"a": {"label_0": 3.0, "label_2": 1.0}}, name="sparse"),
        Run({"a": {"label_1": 2.0, "label_2": 1.5}}, name="dense"),
        {"a": {"label_0": 1.0, "label_2": 1.0}},
    )
    f1 = (
        Run({"b": {"label_0": 2.0, "label_2": 1.2}}, name="sparse"),
        Run({"b": {"label_1": 1.0, "label_2": 1.8}}, name="dense"),
        {"b": {"label_2": 1.0}},
    )
    fold_runs = [f0, f1]
    mcfg = MetricsConfig(ks=(1,), kinds=("ndcg",))
    # um optimize (wsum) e um ponderado (wcondorcet); ambos otimizam p/ cauda
    for method in ("wsum", "wcondorcet"):
        agg = evaluate_combo_supervised(
            "zmuv", method, fold_runs, head={0, 1}, tail={2}, mcfg=mcfg,
            select_segment="tail", select_metric="ndcg@1",
        )
        assert set(agg) == {"overall", "head", "tail"}
        mean, std = agg["tail"]["ndcg@1"]
        assert 0.0 <= mean <= 1.0          # métrica válida, sem NaN (2 folds → há treino)
