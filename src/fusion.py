"""Fusão de rankings (esparso + denso) para XMTC — a contribuição central do projeto.

Combina os dois runs base por fold (BM25 kNN + bi-encoder denso) num único ranking,
varrendo o produto **(normalização × algoritmo de fusão)** que o artigo-base estuda
(França et al. 2025, arXiv 2507.03761): 6 normalizações × 10 fusões = 60 combinações,
melhor reportada = **CombMNZ + ZMUV**.

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
# São as 6 normalizações do artigo (Seção 4).
NORMS: dict[str, str] = {
    "minmax": "min-max",
    "max": "max",
    "sum": "sum",
    "zmuv": "zmuv",
    "rank": "rank",
    "borda": "borda",
}
# São as 10 fusões do artigo (score/rank/voting-based).
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
}


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
    n_folds: int = 5

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
    fused = fuse(runs, norm=NORMS[norm], method=METHODS[method], params=params)
    fused.name = f"{method}_{norm}"
    return fused


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
    params = {"k": cfg.rrf_k} if method == "rrf" else None  # rrf não está no grid do artigo; reservado
    fused = fuse_runs([sparse, dense], norm=norm, method=method, params=params)
    out = cfg.out_path(fold_id, norm, method)
    save_run(fused, out)
    return out


def run_cv(cfg: FusionConfig | None = None) -> None:
    """Funde o par (cfg.norm, cfg.method) — default CombMNZ+ZMUV — nos N folds."""
    cfg = cfg or FusionConfig()
    print(f"fusão: {cfg.method} × {cfg.norm} | {cfg.n_folds} folds")
    for f in range(cfg.n_folds):
        out = fuse_fold(cfg, f, cfg.norm, cfg.method)
        print(f"  fold {f}: {out}")


def run_grid(cfg: FusionConfig | None = None) -> None:
    """Funde TODAS as 60 combinações (6 norm × 10 fusão) em todos os folds.
    Insumo do gridsearch.py (que avaliará cada run fundido com metrics.py)."""
    cfg = cfg or FusionConfig()
    total = len(NORMS) * len(METHODS) * cfg.n_folds
    print(f"grid de fusão: {len(NORMS)} norm × {len(METHODS)} fusão × {cfg.n_folds} folds = {total} runs")
    done = 0
    for norm in NORMS:
        for method in METHODS:
            for f in range(cfg.n_folds):
                fuse_fold(cfg, f, norm, method)
                done += 1
        print(f"  {norm}: {len(METHODS) * cfg.n_folds} runs ({done}/{total})")


def main() -> None:
    cfg = FusionConfig()
    if "--grid" in sys.argv:
        run_grid(cfg)
    else:
        run_cv(cfg)


if __name__ == "__main__":
    main()
