"""Fusão de rankings (esparso + denso) para XMTC — a contribuição central do projeto.

Combina os dois runs base por fold (BM25 kNN + bi-encoder denso) num único ranking,
varrendo o produto **(normalização × algoritmo de fusão)** que o artigo-base estuda
(França et al. 2025, arXiv 2507.03761): 6 normalizações × 10 fusões, melhor reportada
= **CombMNZ + ZMUV**. Estendemos a grade para o conjunto COMPLETO do ranx — 7
normalizações × **25 fusões** = 175 combinações — conforme a proposta do projeto.

As 25 fusões dividem-se em:
  - **14 não-supervisionadas** (paramétrico-fixas): as 10 do artigo + `rrf`,
    `logn_isr`, `rbc` (φ), `gmnz` (γ). Fundem direto os runs de teste.
  - **11 supervisionadas** (aprendem parâmetros num conjunto de treino com qrels):
    `wsum`, `wmnz`, `mixed`, `bayesfuse`, `mapfuse`, `posfuse`, `probfuse`,
    `segfuse`, `slidefuse` (via `ranx.optimize_fusion`) + `w_bordafuse`,
    `w_condorcet` (peso otimizado à mão, pois o ranx não os otimiza). O treino usa
    **CV aninhada** (params do fold k aprendidos nos OUTROS folds — sem vazamento) e
    otimiza para uma **métrica de cauda** (foco da pergunta de pesquisa). Ver
    `learn_fusion_params` aqui e a orquestração em gridsearch.py.

Fonte da verdade = `ranx` (AmenRa/ranx), como manda o guardrail do projeto. Para
demonstrar entendimento e validar o ranx, reimplementamos 2 algoritmos à mão
(**CombMNZ** e **RRF**) em Python puro e os testes conferem que a ORDEM produzida
bate com a do ranx (tests/test_fusion.py).

Formato canônico = run TREC (`qid Q0 label_id rank score tag`), o mesmo do esparso e
do denso. A fusão é OFFLINE e barata: gera-se os runs base UMA vez e itera-se aqui
(o grid completo de seleção, com métricas, fica em gridsearch.py + metrics.py).

Import do ranx é LAZY (dentro das funções), espelhando o import tardio do retriv/vllm
nos recuperadores — as funções à mão (puras) são testáveis sem o ranx.

Uso:
    python -m src.fusion                 # funde o melhor par (CombMNZ+ZMUV) nos 5 folds
    python -m src.fusion --grid          # funde TODAS as 60 combinações em todos os folds
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

if __package__ in {None, ""}:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Mapeamento nome-do-artigo → string aceita pelo ranx 0.3.21 (validado).
# As 6 normalizações do artigo (Seção 4) + min-max-inverted (extra do ranx, para
# scores invertidos/de distância — inverte a ordem; útil p/ ablação, não para o
# melhor resultado, já que nossos scores são similaridades).
NORMS: dict[str, str] = {
    "minmax": "min-max",
    "minmaxinv": "min-max-inverted",
    "max": "max",
    "sum": "sum",
    "zmuv": "zmuv",
    "rank": "rank",
    "borda": "borda",
}
# As 10 fusões do artigo (score/rank/voting-based) + 4 extras do ranx para teste:
# rrf (rank), logn_isr (variante log-n do ISR), rbc (rank-biased centroids, param φ),
# gmnz (CombMNZ generalizado, param γ). Parâmetros default vêm de FusionConfig.
METHODS: dict[str, str] = {
    "combmin": "min",
    "combmax": "max",
    "combmed": "med",
    "combsum": "sum",
    "combanz": "anz",
    "combmnz": "mnz",
    "isr": "isr",
    "logisr": "log_isr",
    "bordafuse": "bordafuse",
    "condorcet": "condorcet",
    # extras não-supervisionados (além do artigo)
    "rrf": "rrf",
    "lognisr": "logn_isr",
    "rbc": "rbc",
    "gmnz": "gmnz",
    # supervisionadas (completam a grade de 25 da proposta) — chave nossa → string ranx
    "wsum": "wsum",
    "wmnz": "wmnz",
    "mixed": "mixed",
    "bayesfuse": "bayesfuse",
    "mapfuse": "mapfuse",
    "posfuse": "posfuse",
    "probfuse": "probfuse",
    "segfuse": "segfuse",
    "slidefuse": "slidefuse",
    "wbordafuse": "w_bordafuse",
    "wcondorcet": "w_condorcet",
}

# Métodos que exigem parâmetro extra no ranx → como mapear de FusionConfig.
_PARAMIZED = {"rrf", "rbc", "gmnz"}

# Fusões SUPERVISIONADAS: aprendem parâmetros num conjunto de treino (qrels). Duas
# famílias, pelo mecanismo do ranx:
#  - SUPERVISED_OPTIMIZE: `ranx.optimize_fusion` aprende os params (pesos de wsum/
#    wmnz/mixed; distribuições dos métodos `*_train`).
#  - SUPERVISED_WEIGHTED: o ranx NÃO otimiza (w_bordafuse/w_condorcet); otimizamos o
#    peso à mão (1 escalar p/ 2 runs) maximizando a métrica-alvo no treino.
SUPERVISED_OPTIMIZE: set[str] = {
    "wsum", "wmnz", "mixed", "bayesfuse", "mapfuse",
    "posfuse", "probfuse", "segfuse", "slidefuse",
}
SUPERVISED_WEIGHTED: set[str] = {"wbordafuse", "wcondorcet"}
SUPERVISED: set[str] = SUPERVISED_OPTIMIZE | SUPERVISED_WEIGHTED


def method_params(method: str, k: int = 60, phi: float = 0.8, gamma: float = 2.0) -> dict | None:
    """Parâmetros default por método (None para os sem parâmetro). As chaves são as
    que o ranx 0.3.21 espera: rrf→k, rbc→phi, gmnz→gamma."""
    return {
        "rrf": {"k": k},
        "rbc": {"phi": phi},
        "gmnz": {"gamma": gamma},
    }.get(method)


@dataclass
class FusionConfig:
    runs_dir: str = "data/eurlex4k/runs"
    sparse_template: str = "sparse.fold{fold}.trec"
    dense_template: str = "dense.fold{fold}.trec"
    out_template: str = "fused.{norm}.{method}.fold{fold}.trec"
    # par default = melhor do artigo (CombMNZ + ZMUV). Chaves nos dicts NORMS/METHODS.
    norm: str = "zmuv"
    method: str = "combmnz"
    rrf_k: int = 60            # k do RRF (default do ranx)
    rbc_phi: float = 0.8       # φ do RBC (rank-biased centroids)
    gmnz_gamma: float = 2.0    # γ do GMNZ (CombMNZ generalizado)
    n_folds: int = 5
    folds: tuple[int, ...] | None = None   # None = todos; subconjunto p/ "3 dos 5 folds"

    def fold_ids(self) -> tuple[int, ...]:
        return self.folds if self.folds is not None else tuple(range(self.n_folds))

    def sparse_path(self, fold_id: int) -> str:
        return os.path.join(self.runs_dir, self.sparse_template.format(fold=fold_id))

    def dense_path(self, fold_id: int) -> str:
        return os.path.join(self.runs_dir, self.dense_template.format(fold=fold_id))

    def out_path(self, fold_id: int, norm: str, method: str) -> str:
        return os.path.join(self.runs_dir, self.out_template.format(fold=fold_id, norm=norm, method=method))


# ─────────────────────────── wrappers do ranx (fonte da verdade) ───────────────────

def load_run(path: str, name: str | None = None):
    """Lê um run TREC como ranx.Run (import lazy do ranx)."""
    from ranx import Run

    if not os.path.exists(path):
        raise FileNotFoundError(f"run não encontrado: {path}")
    run = Run.from_file(path, kind="trec")
    if name:
        run.name = name
    return run


def _align_to_common_qids(runs: list) -> list:
    """Restringe os runs ao conjunto COMUM de qids. `ranx.fuse` exige que todos os
    runs tenham EXATAMENTE os mesmos qids; mas um doc de teste pode faltar num
    recuperador — ex.: texto vazio (alguns artigos do Wiki10 vêm em branco) → BM25
    sem vizinhos → qid ausente no run esparso, mas presente no denso (o embedding
    sempre existe). Fundimos só onde os DOIS têm sinal (interseção); os poucos qids
    descartados são docs vazios (~0,02% no Wiki10), sem sinal esparso pra fundir. Se
    os conjuntos já batem (caso comum), devolve os runs intactos, sem custo."""
    from ranx import Run

    key_sets = [set(r.keys()) for r in runs]
    common = set.intersection(*key_sets)
    if all(len(ks) == len(common) for ks in key_sets):
        return runs
    aligned = []
    for r in runs:
        nr = Run({q: scores for q, scores in r.to_dict().items() if q in common})
        nr.name = r.name
        aligned.append(nr)
    return aligned


def fuse_runs(runs: list, norm: str, method: str, params: dict | None = None):
    """Funde uma lista de ranx.Run com (normalização × fusão), via ranx.

    `norm`/`method` são as CHAVES dos dicts NORMS/METHODS (nomes do artigo); a
    tradução para a string do ranx acontece aqui. Retorna um ranx.Run.
    """
    from ranx import fuse

    if norm not in NORMS:
        raise ValueError(f"normalização {norm!r} desconhecida; use uma de {sorted(NORMS)}.")
    if method not in METHODS:
        raise ValueError(f"fusão {method!r} desconhecida; use uma de {sorted(METHODS)}.")
    runs = _align_to_common_qids(runs)        # ranx.fuse exige qids idênticos entre runs
    fused = fuse(runs, norm=NORMS[norm], method=METHODS[method], params=params)
    fused.name = f"{method}_{norm}"
    return fused


_WEIGHT_GRID = tuple((i / 10, 1 - i / 10) for i in range(11))   # (1.0,0.0)..(0.0,1.0)


def _fit_params_once(qrels, runs, norm: str, method: str, metric: str) -> dict:
    """Aprende os params UMA vez no (qrels, runs) dado: `optimize_fusion` para as
    SUPERVISED_OPTIMIZE; grid-search de peso para as SUPERVISED_WEIGHTED."""
    from ranx import fuse, optimize_fusion
    from ranx import evaluate as ranx_evaluate

    if method in SUPERVISED_OPTIMIZE:
        return optimize_fusion(
            qrels=qrels, runs=runs, norm=NORMS[norm], method=METHODS[method], metric=metric,
        )
    best_w, best_score = _WEIGHT_GRID[5], -1.0
    for w in _WEIGHT_GRID:
        fused = fuse(runs, norm=NORMS[norm], method=METHODS[method], params={"weights": w})
        score = float(ranx_evaluate(qrels, fused, metric))
        if score > best_score:
            best_score, best_w = score, w
    return {"weights": best_w}


def learn_fusion_params(
    train_qrels: dict,
    train_runs: list,
    norm: str,
    method: str,
    metric: str = "ndcg@5",
    sample_size: int = 0,
    repeats: int = 1,
) -> dict:
    """Aprende os parâmetros de uma fusão SUPERVISIONADA num conjunto de treino, via
    CV aninhada (`train_*` são os folds de TREINO, disjuntos do teste). `train_qrels`
    vem restrito ao alvo (cauda) para mirar a métrica de cauda.

    Aceleração (datasets grandes) por SUBAMOSTRAGEM do treino da otimização: os params
    (nº de segmentos, pesos) são de baixa dimensão → uma amostra de `sample_size`
    queries estima-os quase igual ao treino cheio, mas o `optimize_fusion` fica
    ~N/sample× mais leve. A AVALIAÇÃO final NÃO usa amostra — roda nos folds de teste
    completos (gridsearch.py); só o aprendizado do parâmetro usa a amostra.

    Validade científica (`repeats>1`): em vez de UMA amostra (sujeita a variância de
    sorteio), tira `repeats` amostras independentes → `repeats` candidatos → e SELECIONA
    o que pontua melhor **no treino COMPLETO** (re-avaliação barata: 1 fuse+eval por
    candidato). Uniforme p/ os 11 métodos (não precisa "mediar" o parâmetro, só testar
    e escolher) e principiada (o vencedor generaliza melhor ao treino inteiro, não a uma
    amostra com sorte). Custo ≈ repeats × (otimização na amostra) + repeats × (1 eval
    no cheio) → manter `repeats × sample_size` bem abaixo do treino total preserva o
    ganho de tempo.
    """
    from ranx import Qrels, Run

    if method not in SUPERVISED:
        raise ValueError(f"{method!r} não é supervisionada (use SUPERVISED).")
    if not train_qrels:
        raise ValueError("train_qrels vazio — sem sinal de treino (ex.: nenhum gold de cauda).")

    # ranx exige run.keys() == qrels.keys() EXATAMENTE. Os qrels de cauda descartam
    # queries sem gold de cauda → alinha tudo ao conjunto COMUM (qrels ∩ todos os runs).
    common = set(train_qrels)
    for r in train_runs:
        common &= set(r.keys())
    if not common:
        raise ValueError("sem qids em comum entre qrels (cauda) e runs de treino.")

    def build(qids):
        """(Qrels, [Run]) restritos a `qids`."""
        q = Qrels({x: train_qrels[x] for x in qids})
        rs = []
        for r in train_runs:
            nr = Run({x: s for x, s in r.to_dict().items() if x in qids})
            nr.name = r.name
            rs.append(nr)
        return q, rs

    # sem subamostragem (ou amostra ≥ treino): caminho direto no conjunto cheio
    if not sample_size or len(common) <= sample_size:
        q, rs = build(common)
        return _fit_params_once(q, rs, norm, method, metric)

    import random
    sorted_common = sorted(common)

    # 1 amostra só (repeats=1): cap simples
    if repeats <= 1:
        sub = random.Random(42).sample(sorted_common, sample_size)
        q, rs = build(sub)
        return _fit_params_once(q, rs, norm, method, metric)

    # K restarts + SELEÇÃO no treino completo (robusto à variância de amostra)
    from ranx import fuse
    from ranx import evaluate as ranx_evaluate

    qf, rsf = build(common)                     # treino completo, para pontuar candidatos
    best_params, best_score = None, float("-inf")
    for i in range(repeats):
        sub = random.Random(42 + i).sample(sorted_common, sample_size)
        q, rs = build(sub)
        cand = _fit_params_once(q, rs, norm, method, metric)
        fused = fuse(rsf, norm=NORMS[norm], method=METHODS[method], params=cand)
        score = float(ranx_evaluate(qf, fused, metric))
        if score > best_score:
            best_score, best_params = score, cand
    return best_params


def save_run(run, path: str) -> None:
    """Grava um ranx.Run em TREC (cria o diretório-pai)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    run.save(path, kind="trec")


# ─────────────────── implementações à mão (validadas contra o ranx) ────────────────
# Operam sobre dict-of-dicts {qid: {label: score}} — Python puro, sem ranx.

def run_to_dict(run) -> dict[str, dict[str, float]]:
    """ranx.Run → {qid: {label: score}} com floats nativos (p/ as versões à mão)."""
    return {q: {d: float(s) for d, s in docs.items()} for q, docs in run.to_dict().items()}


def _min_max_query(scores: dict[str, float]) -> dict[str, float]:
    """Min-Max por query: (s - min)/(max - min). Tudo igual → todos 1.0 (evita /0)."""
    if not scores:
        return {}
    vals = scores.values()
    lo, hi = min(vals), max(vals)
    if hi == lo:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def comb_mnz(run_dicts: list[dict[str, dict[str, float]]], normalize: bool = True) -> dict[str, dict[str, float]]:
    """CombMNZ à mão: score = (Σ scores normalizados nos runs em que o rótulo aparece)
    × (nº de runs em que aparece). Com `normalize=True` aplica Min-Max por query antes
    de somar (réplica do que o ranx faz com norm='min-max').

    É a melhor família do artigo; reimplementada para demonstrar entendimento e servir
    de checagem contra o ranx (ver tests/test_fusion.py)."""
    qids: set[str] = set()
    for rd in run_dicts:
        qids.update(rd.keys())

    out: dict[str, dict[str, float]] = {}
    for q in qids:
        per_run = [_min_max_query(rd.get(q, {})) if normalize else dict(rd.get(q, {})) for rd in run_dicts]
        sums: dict[str, float] = {}
        hits: dict[str, int] = {}
        for scores in per_run:
            for label, s in scores.items():
                sums[label] = sums.get(label, 0.0) + s
                hits[label] = hits.get(label, 0) + 1
        out[q] = {label: sums[label] * hits[label] for label in sums}
    return out


def rrf(run_dicts: list[dict[str, dict[str, float]]], k: int = 60) -> dict[str, dict[str, float]]:
    """Reciprocal Rank Fusion à mão: score = Σ_runs 1/(k + rank). Baseado só em posição
    (não usa o valor do score), então é robusto a escalas díspares — daí dispensar
    normalização. Empates resolvidos por ordenação estável do score."""
    qids: set[str] = set()
    for rd in run_dicts:
        qids.update(rd.keys())

    out: dict[str, dict[str, float]] = {}
    for q in qids:
        fused: dict[str, float] = {}
        for rd in run_dicts:
            ranked = sorted(rd.get(q, {}).items(), key=lambda kv: kv[1], reverse=True)
            for rank, (label, _score) in enumerate(ranked, start=1):
                fused[label] = fused.get(label, 0.0) + 1.0 / (k + rank)
        out[q] = fused
    return out


def ranking(run_dict: dict[str, dict[str, float]]) -> dict[str, list[str]]:
    """{qid: [labels ordenados por score desc]} — usado p/ comparar ORDEM com o ranx."""
    return {
        q: [label for label, _ in sorted(docs.items(), key=lambda kv: kv[1], reverse=True)]
        for q, docs in run_dict.items()
    }


# ─────────────────────────── orquestração por fold ─────────────────────────────────

def fuse_fold(cfg: FusionConfig, fold_id: int, norm: str, method: str) -> str:
    """Lê sparse/dense do fold, funde com (norm, method) e grava o run fundido.
    Retorna o caminho de saída."""
    sparse = load_run(cfg.sparse_path(fold_id), name="sparse")
    dense = load_run(cfg.dense_path(fold_id), name="dense")
    params = method_params(method, k=cfg.rrf_k, phi=cfg.rbc_phi, gamma=cfg.gmnz_gamma)
    fused = fuse_runs([sparse, dense], norm=norm, method=method, params=params)
    out = cfg.out_path(fold_id, norm, method)
    save_run(fused, out)
    return out


def run_cv(cfg: FusionConfig | None = None) -> None:
    """Funde o par (cfg.norm, cfg.method) — default CombMNZ+ZMUV — nos N folds."""
    cfg = cfg or FusionConfig()
    fold_ids = cfg.fold_ids()
    print(f"fusão: {cfg.method} × {cfg.norm} | folds {list(fold_ids)}")
    for f in fold_ids:
        out = fuse_fold(cfg, f, cfg.norm, cfg.method)
        print(f"  fold {f}: {out}")


def run_grid(cfg: FusionConfig | None = None) -> None:
    """Funde TODAS as 60 combinações (6 norm × 10 fusão) em todos os folds.
    Insumo do gridsearch.py (que avaliará cada run fundido com metrics.py)."""
    cfg = cfg or FusionConfig()
    fold_ids = cfg.fold_ids()
    total = len(NORMS) * len(METHODS) * len(fold_ids)
    print(f"grid de fusão: {len(NORMS)} norm × {len(METHODS)} fusão × {len(fold_ids)} folds = {total} runs")
    done = 0
    for norm in NORMS:
        for method in METHODS:
            for f in fold_ids:
                fuse_fold(cfg, f, norm, method)
                done += 1
        print(f"  {norm}: {len(METHODS) * len(fold_ids)} runs ({done}/{total})")


def main() -> None:
    import argparse

    from src.data import add_dataset_arg, add_folds_arg, apply_dataset, parse_folds

    parser = argparse.ArgumentParser(description="Fusão dos runs base (norm × método)")
    add_dataset_arg(parser)
    add_folds_arg(parser)
    parser.add_argument("--grid", action="store_true", help="funde as combinações do grid")
    args, _ = parser.parse_known_args()

    cfg = FusionConfig()
    apply_dataset(cfg, args.dataset)
    cfg.folds = parse_folds(args.folds)
    if args.grid:
        run_grid(cfg)
    else:
        run_cv(cfg)


if __name__ == "__main__":
    main()
