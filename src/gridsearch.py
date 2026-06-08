"""Grid search da fusão — a contribuição central: qual (normalização × fusão)
melhor recupera *tail labels* sem prejudicar a cabeça?

Varre as combinações de fusion.NORMS × fusion.METHODS (98 por padrão: 7 norm × 14
fusão), funde os runs base esparso+denso por fold EM MEMÓRIA (os runs base são
carregados UMA vez e reusados — fusão é offline e barata), avalia cada uma com as
métricas segmentadas cabeça/cauda de metrics.py, agrega entre folds e ranqueia por
uma métrica de cauda escolhida (default: tail nDCG@5).

Saída: ranking impresso (top-N + onde cai o CombMNZ+ZMUV do artigo) e um CSV
long-format com TODAS as combinações × segmentos × métricas (insumo do relatório).

Reúsa fusion.py (fusão) e metrics.py (avaliação) — nada reimplementado aqui.

Uso:
    python -m src.gridsearch                       # 98 combos, ranqueia por tail nDCG@5
    python -m src.gridsearch --select tail:precision@5
    python -m src.gridsearch --paper               # só as 6×10 do artigo
"""
from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass, field

if __package__ in {None, ""}:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fusion import METHODS, NORMS, fuse_runs, load_run, method_params, run_to_dict
from src.metrics import (
    SEGMENTS,
    MetricsConfig,
    aggregate,
    build_qrels,
    evaluate_segmented,
    metric_names,
)
from src.retrieve_sparse import head_tail_split
from src.splits import load_pooled

# combos do artigo (6 norm × 10 fusão) — p/ o modo --paper
PAPER_NORMS = ("minmax", "max", "sum", "zmuv", "rank", "borda")
PAPER_METHODS = ("combmin", "combmax", "combmed", "combsum", "combanz", "combmnz",
                 "isr", "logisr", "bordafuse", "condorcet")


@dataclass
class GridConfig:
    raw_dir: str = "data/eurlex4k/raw"
    runs_dir: str = "data/eurlex4k/runs"
    sparse_template: str = "sparse.fold{fold}.trec"
    dense_template: str = "dense.fold{fold}.trec"
    ks: tuple[int, ...] = (1, 5, 10)
    kinds: tuple[str, ...] = ("precision", "ndcg", "recall")
    head_frac: float = 0.20
    n_folds: int = 5
    # parâmetros das fusões parametrizadas (= FusionConfig)
    rrf_k: int = 60
    rbc_phi: float = 0.8
    gmnz_gamma: float = 2.0
    # seleção do ranking
    select_segment: str = "tail"       # a RQ é sobre a cauda
    select_metric: str = "ndcg@5"
    top_n: int = 12
    out_csv: str = "data/eurlex4k/results/gridsearch.csv"
    norms: tuple[str, ...] = field(default_factory=lambda: tuple(NORMS))
    methods: tuple[str, ...] = field(default_factory=lambda: tuple(METHODS))

    def metrics_config(self) -> MetricsConfig:
        return MetricsConfig(ks=self.ks, kinds=self.kinds, head_frac=self.head_frac, n_folds=self.n_folds)


# ─────────────────────── avaliação de UMA combinação (testável) ────────────────────

def evaluate_combo(
    norm: str,
    method: str,
    fold_runs: list[tuple],
    head: set[int],
    tail: set[int],
    mcfg: MetricsConfig,
    params: dict | None = None,
) -> dict[str, dict[str, tuple[float, float]]]:
    """Funde (norm × method) em cada fold e agrega as métricas segmentadas.

    `fold_runs` = lista de (sparse_run, dense_run, qrels_fold) por fold — runs são
    objetos ranx.Run (carregados uma vez e reusados por todas as combinações).
    Retorna {segmento: {métrica: (média, desvio)}}.
    """
    per_fold: dict[str, list[dict]] = {s: [] for s in SEGMENTS}
    for sparse_run, dense_run, qrels_fold in fold_runs:
        fused = fuse_runs([sparse_run, dense_run], norm=norm, method=method, params=params)
        seg = evaluate_segmented(run_to_dict(fused), qrels_fold, head, tail, mcfg)
        for s in SEGMENTS:
            per_fold[s].append(seg[s])
    return {s: aggregate(per_fold[s]) for s in SEGMENTS}


def rank_records(
    records: list[dict], segment: str, metric: str, descending: bool = True
) -> list[dict]:
    """Ordena os registros pela média da (segmento, métrica) escolhida."""
    return sorted(
        records,
        key=lambda r: r["agg"][segment][metric][0],
        reverse=descending,
    )


# ─────────────────────────── orquestração (carrega disco) ──────────────────────────

def load_fold_runs(cfg: GridConfig) -> tuple[list[tuple], set[int], set[int]]:
    """Carrega os runs base esparso+denso de cada fold (uma vez) + qrels do fold, e
    devolve também os conjuntos cabeça/cauda globais. Falha se faltar algum run."""
    pooled = load_pooled(cfg.raw_dir)
    head, tail = head_tail_split(pooled.label_cols, pooled.n_labels, cfg.head_frac)
    qrels_all = build_qrels(pooled)

    fold_runs: list[tuple] = []
    for f in range(cfg.n_folds):
        sp = os.path.join(cfg.runs_dir, cfg.sparse_template.format(fold=f))
        de = os.path.join(cfg.runs_dir, cfg.dense_template.format(fold=f))
        for p in (sp, de):
            if not os.path.exists(p):
                raise FileNotFoundError(f"run base ausente: {p} (rode retrieve_sparse/retrieve_dense)")
        sparse_run = load_run(sp, name="sparse")
        dense_run = load_run(de, name="dense")
        qids = run_to_dict(sparse_run).keys()
        qrels_fold = {q: qrels_all[q] for q in qids if q in qrels_all}
        fold_runs.append((sparse_run, dense_run, qrels_fold))
    return fold_runs, head, tail


def run_grid(cfg: GridConfig | None = None) -> list[dict]:
    """Avalia todas as combinações (norms × methods) e retorna os registros
    [{norm, method, agg}], ordenados pela métrica de seleção."""
    cfg = cfg or GridConfig()
    mcfg = cfg.metrics_config()
    fold_runs, head, tail = load_fold_runs(cfg)
    total = len(cfg.norms) * len(cfg.methods)
    print(
        f"grid: {len(cfg.norms)} norm × {len(cfg.methods)} fusão = {total} combos "
        f"| {cfg.n_folds} folds | ranqueando por {cfg.select_segment} {cfg.select_metric}"
    )

    records: list[dict] = []
    done = 0
    for method in cfg.methods:
        params = method_params(method, k=cfg.rrf_k, phi=cfg.rbc_phi, gamma=cfg.gmnz_gamma)
        for norm in cfg.norms:
            agg = evaluate_combo(norm, method, fold_runs, head, tail, mcfg, params)
            records.append({"norm": norm, "method": method, "agg": agg})
            done += 1
        print(f"  {method}: {len(cfg.norms)} combos ({done}/{total})")

    return rank_records(records, cfg.select_segment, cfg.select_metric)


# ─────────────────────────── relatório + CSV ───────────────────────────────────────

def format_grid_report(ranked: list[dict], cfg: GridConfig) -> str:
    """Top-N por (segmento, métrica) de seleção; mostra cauda + cabeça lado a lado
    pra evidenciar 'melhora a cauda sem prejudicar a cabeça'."""
    seg, met = cfg.select_segment, cfg.select_metric
    cols = [("tail", met), ("tail", "precision@1"), ("tail", "precision@5"),
            ("head", met), ("overall", met)]
    head_row = f"{'#':>2}  {'fusão+norma':<22} " + " ".join(f"{s[:1]}:{m:<11}" for s, m in cols)
    lines = [f"\nRanking por {seg} {met} (top {cfg.top_n} de {len(ranked)}):", head_row, "-" * len(head_row)]
    for i, r in enumerate(ranked[:cfg.top_n], 1):
        combo = f"{r['method']}+{r['norm']}"
        cells = " ".join(f"{r['agg'][s][m][0]:<13.4f}" for s, m in cols)
        lines.append(f"{i:>2}  {combo:<22} {cells}")
    # onde caiu o par do artigo (CombMNZ+ZMUV)
    for i, r in enumerate(ranked, 1):
        if r["method"] == "combmnz" and r["norm"] == "zmuv":
            lines.append(f"\nReferência do artigo (CombMNZ+ZMUV): #{i} em {seg} {met} "
                         f"= {r['agg'][seg][met][0]:.4f}")
            break
    return "\n".join(lines)


def save_csv(ranked: list[dict], cfg: GridConfig) -> None:
    """CSV long-format: norm, method, segment, metric, mean, std — todas as combos."""
    os.makedirs(os.path.dirname(cfg.out_csv), exist_ok=True)
    names = metric_names(cfg.ks, cfg.kinds)
    with open(cfg.out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "norm", "segment", "metric", "mean", "std"])
        for r in ranked:
            for seg in SEGMENTS:
                for n in names:
                    mean, std = r["agg"][seg][n]
                    w.writerow([r["method"], r["norm"], seg, n, f"{mean:.6f}", f"{std:.6f}"])
    print(f"\nCSV salvo: {cfg.out_csv} ({len(ranked)} combos × {len(SEGMENTS)} segmentos × {len(names)} métricas)")


def main(cfg: GridConfig | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Grid search da fusão (norm × método)")
    parser.add_argument("--select", type=str, default=None,
                        help="métrica de seleção no formato segmento:metrica (ex.: tail:ndcg@5)")
    parser.add_argument("--paper", action="store_true",
                        help="só as 6×10 combinações do artigo (exclui rrf/rbc/gmnz/logn_isr/min-max-inverted)")
    parser.add_argument("--top", type=int, default=None, help="quantas combos imprimir")
    args, _ = parser.parse_known_args()

    cfg = cfg or GridConfig()
    if args.paper:
        cfg.norms, cfg.methods = PAPER_NORMS, PAPER_METHODS
    if args.select:
        seg, met = args.select.split(":", 1)
        cfg.select_segment, cfg.select_metric = seg, met
    if args.top:
        cfg.top_n = args.top

    ranked = run_grid(cfg)
    print(format_grid_report(ranked, cfg))
    save_csv(ranked, cfg)


if __name__ == "__main__":
    main()
