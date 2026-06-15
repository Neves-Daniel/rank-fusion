"""Testes de src/fusion.py.

Duas camadas:
  - Funções À MÃO (CombMNZ, RRF) em Python puro: testes determinísticos, sem ranx —
    incluindo casos calculáveis na mão.
  - Validação contra o RANX (fonte da verdade): confere que a ORDEM dos rótulos
    produzida à mão bate com a do ranx, e que as 60 combinações (6 norm × 10 fusão)
    do artigo rodam + round-trip TREC. Pulado se o ranx não estiver instalado.

Importar src.fusion NÃO carrega o ranx (import lazy dentro das funções).
"""
import subprocess
import sys

import pytest

from src.fusion import (
    METHODS,
    NORMS,
    FusionConfig,
    comb_mnz,
    method_params,
    ranking,
    rrf,
)


def test_import_nao_puxa_ranx():
    code = (
        "import sys, src.fusion; "
        "assert 'ranx' not in sys.modules, 'ranx importado no load de src.fusion'; "
        "print('ok')"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert "ok" in res.stdout


def test_mapeamentos_artigo_mais_extras():
    # 6 norm + min-max-inverted = 7 ; grade COMPLETA do ranx = 25 fusões
    # (14 não-supervisionadas + 11 supervisionadas), conforme a proposta.
    from src.fusion import SUPERVISED, SUPERVISED_OPTIMIZE, SUPERVISED_WEIGHTED

    assert len(NORMS) == 7 and len(METHODS) == 25
    assert NORMS["minmaxinv"] == "min-max-inverted"
    assert {"rrf", "lognisr", "rbc", "gmnz"} <= set(METHODS)
    assert NORMS["zmuv"] == "zmuv" and METHODS["combmnz"] == "mnz"
    # 11 supervisionadas: 9 via optimize + 2 ponderadas; disjuntas das 14 fixas
    assert len(SUPERVISED) == 11 and len(SUPERVISED_OPTIMIZE) == 9 and len(SUPERVISED_WEIGHTED) == 2
    assert SUPERVISED <= set(METHODS)
    assert METHODS["wbordafuse"] == "w_bordafuse" and METHODS["wsum"] == "wsum"
    assert len(set(METHODS) - SUPERVISED) == 14   # as 14 não-supervisionadas


def test_method_params_so_para_parametrizados():
    assert method_params("combmnz") is None
    assert method_params("rrf", k=42) == {"k": 42}
    assert method_params("rbc", phi=0.9) == {"phi": 0.9}
    assert method_params("gmnz", gamma=3) == {"gamma": 3}


def test_fusion_config_paths():
    cfg = FusionConfig()
    assert cfg.sparse_path(2).endswith("runs/sparse.fold2.trec")
    assert cfg.dense_path(2).endswith("runs/dense.fold2.trec")
    assert cfg.out_path(1, "zmuv", "combmnz").endswith("runs/fused.zmuv.combmnz.fold1.trec")


# ─────────────────────────── CombMNZ à mão (calculável) ────────────────────────────

def test_comb_mnz_sem_normalizar_conta_hits():
    # query q: labelA em ambos os runs (hit=2), labelB só num (hit=1)
    r1 = {"q": {"A": 2.0, "B": 4.0}}
    r2 = {"q": {"A": 6.0}}
    out = comb_mnz([r1, r2], normalize=False)
    # A: (2+6)*2 = 16 ; B: 4*1 = 4
    assert out["q"]["A"] == 16.0
    assert out["q"]["B"] == 4.0


def test_comb_mnz_com_minmax_por_query():
    # run1: A=10,B=0 -> minmax A=1,B=0 ; run2: A=0,C=5 -> minmax A=0,C=1
    r1 = {"q": {"A": 10.0, "B": 0.0}}
    r2 = {"q": {"A": 0.0, "C": 5.0}}
    out = comb_mnz([r1, r2], normalize=True)
    # A: (1+0)*2=2 ; B: 0*1=0 ; C: 1*1=1
    assert out["q"]["A"] == pytest.approx(2.0)
    assert out["q"]["B"] == pytest.approx(0.0)
    assert out["q"]["C"] == pytest.approx(1.0)


def test_comb_mnz_minmax_todos_iguais_nao_divide_por_zero():
    out = comb_mnz([{"q": {"A": 3.0, "B": 3.0}}], normalize=True)
    assert out["q"] == {"A": 1.0, "B": 1.0}   # max==min → todos 1.0


# ─────────────────────────── RRF à mão (calculável) ────────────────────────────────

def test_rrf_soma_reciprocos_de_rank():
    # run1 ordena A>B ; run2 ordena B>A ; k=60
    r1 = {"q": {"A": 9.0, "B": 1.0}}
    r2 = {"q": {"B": 9.0, "A": 1.0}}
    out = rrf([r1, r2], k=60)
    # A: 1/61 (rank1 em r1) + 1/62 (rank2 em r2) ; B: simétrico → empatam
    assert out["q"]["A"] == pytest.approx(1 / 61 + 1 / 62)
    assert out["q"]["B"] == pytest.approx(1 / 61 + 1 / 62)


def test_rrf_item_melhor_rankeado_vence():
    r1 = {"q": {"A": 9.0, "B": 8.0, "C": 1.0}}
    r2 = {"q": {"A": 9.0, "B": 1.0}}
    out = rrf([r1, r2], k=10)
    # A aparece no topo dos dois; deve ter o maior score fundido
    assert max(out["q"], key=out["q"].get) == "A"


def test_ranking_ordena_por_score_desc():
    assert ranking({"q": {"A": 1.0, "B": 3.0, "C": 2.0}}) == {"q": ["B", "C", "A"]}


# ─────────────────────── validação contra o ranx (fonte da verdade) ────────────────

@pytest.fixture
def toy_runs():
    """2 runs de brinquedo (sem empates de score por query) p/ ordem determinística."""
    pytest.importorskip("ranx")
    from ranx import Run

    r1 = Run({"q1": {"A": 3.0, "B": 1.0, "C": 0.5}, "q2": {"A": 2.0, "D": 1.0}}, name="sparse")
    r2 = Run({"q1": {"B": 2.0, "C": 1.5, "E": 0.2}, "q2": {"D": 3.0, "A": 0.1}}, name="dense")
    return r1, r2


def test_todas_as_nao_supervisionadas_rodam_no_ranx(toy_runs):
    # as 14 NÃO-supervisionadas fundem direto (sem treino): 7 norm × 14 = 98 combos
    from src.fusion import SUPERVISED, fuse_runs

    r1, r2 = toy_runs
    nao_sup = [m for m in METHODS if m not in SUPERVISED]
    assert len(nao_sup) == 14
    n = 0
    for norm in NORMS:
        for method in nao_sup:
            params = method_params(method)          # passa φ/γ/k aos parametrizados
            fused = fuse_runs([r1, r2], norm=norm, method=method, params=params)
            assert fused.to_dict()                  # produz ranking não-vazio
            n += 1
    assert n == 98                                  # 7 norm × 14 fusão não-supervisionada


def test_learn_fusion_params_optimize_e_weighted(toy_runs):
    # supervisionadas precisam aprender params; testa um optimize (wsum) e um
    # ponderado (wbordafuse) — ambos devem devolver dict de params usável no fuse.
    from src.fusion import SUPERVISED, fuse_runs, learn_fusion_params

    r1, r2 = toy_runs
    # qrels de treino: usa os próprios candidatos do toy como "gold" (sinal qualquer)
    d1 = r1.to_dict()
    qrels = {q: {next(iter(labels)): 1.0} for q, labels in d1.items() if labels}
    for method in ("wsum", "wbordafuse"):
        assert method in SUPERVISED
        params = learn_fusion_params(qrels, [r1, r2], norm="minmax", method=method, metric="ndcg@5")
        assert "weights" in params and len(params["weights"]) == 2
        fused = fuse_runs([r1, r2], norm="minmax", method=method, params=params)
        assert fused.to_dict()


def test_learn_fusion_params_subamostragem(toy_runs):
    # sample_size < nº de queries de treino → otimiza numa amostra (determinística),
    # ainda devolve params válidos e usáveis no fuse.
    from src.fusion import fuse_runs, learn_fusion_params

    r1, r2 = toy_runs
    qrels = {q: {next(iter(labels)): 1.0} for q, labels in r1.to_dict().items() if labels}
    assert len(qrels) >= 2
    p_full = learn_fusion_params(qrels, [r1, r2], norm="minmax", method="wsum", metric="ndcg@5")
    p_amos = learn_fusion_params(qrels, [r1, r2], norm="minmax", method="wsum",
                                 metric="ndcg@5", sample_size=1)
    assert "weights" in p_amos and len(p_amos["weights"]) == 2
    # determinístico: mesma amostra (seed fixa) → mesmo resultado em 2 chamadas
    p_amos2 = learn_fusion_params(qrels, [r1, r2], norm="minmax", method="wsum",
                                  metric="ndcg@5", sample_size=1)
    assert p_amos == p_amos2
    fused = fuse_runs([r1, r2], norm="minmax", method="wsum", params=p_amos)
    assert fused.to_dict()


def test_learn_fusion_params_k_restarts(toy_runs):
    # opt (2): K amostras + seleção no treino completo → params válidos e determinístico.
    from src.fusion import fuse_runs, learn_fusion_params

    r1, r2 = toy_runs
    qrels = {q: {next(iter(labels)): 1.0} for q, labels in r1.to_dict().items() if labels}
    p = learn_fusion_params(qrels, [r1, r2], norm="minmax", method="wsum",
                            metric="ndcg@5", sample_size=1, repeats=3)
    assert "weights" in p and len(p["weights"]) == 2
    p2 = learn_fusion_params(qrels, [r1, r2], norm="minmax", method="wsum",
                             metric="ndcg@5", sample_size=1, repeats=3)
    assert p == p2                                   # seeds fixas → determinístico
    assert fuse_runs([r1, r2], norm="minmax", method="wsum", params=p).to_dict()


def test_comb_mnz_a_mao_bate_com_ranx(toy_runs):
    from src.fusion import fuse_runs, run_to_dict

    r1, r2 = toy_runs
    mine = comb_mnz([run_to_dict(r1), run_to_dict(r2)], normalize=True)
    ranx_fused = fuse_runs([r1, r2], norm="minmax", method="combmnz")
    # mesma ORDEM de rótulos por query (scores podem diferir por detalhes de escala)
    assert ranking(mine) == ranking(run_to_dict(ranx_fused))


def test_rrf_a_mao_bate_com_ranx(toy_runs):
    from src.fusion import fuse_runs, run_to_dict

    r1, r2 = toy_runs
    mine = rrf([run_to_dict(r1), run_to_dict(r2)], k=60)
    ranx_fused = fuse_runs([r1, r2], norm="minmax", method="combmnz", params=None)
    # compara contra o RRF do ranx (não o mnz): usa o método nativo via fuse
    from ranx import fuse

    ranx_rrf = fuse([r1, r2], norm="min-max", method="rrf", params={"k": 60})
    assert ranking(mine) == ranking(run_to_dict(ranx_rrf))


def test_trec_round_trip(tmp_path, toy_runs):
    from src.fusion import fuse_runs, load_run, save_run

    r1, r2 = toy_runs
    fused = fuse_runs([r1, r2], norm="zmuv", method="combmnz")
    out = tmp_path / "fused.zmuv.combmnz.fold0.trec"
    save_run(fused, str(out))
    assert out.exists()
    reloaded = load_run(str(out))
    assert set(reloaded.to_dict().keys()) == {"q1", "q2"}
