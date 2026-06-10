"""Avaliação dos runs (base e fundidos) — com métricas SEGMENTADAS cabeça/cauda.

A pergunta de pesquisa é sobre *tail labels*, então nunca reportamos só métricas
globais (dominadas pela cabeça). Para cada run computamos P@k / nDCG@k / Recall@k em
três recortes:
  - **overall:** o ranking inteiro vs todos os rótulos-gold;
  - **head:** restringindo ranking E gold aos rótulos de CABEÇA (Pareto 80/20 global);
  - **tail:** idem para os rótulos de CAUDA.

É o split cabeça/cauda exigido pelo guardrail do projeto (e o que o artigo-base
reporta nas tabelas). PSP@k/PSnDCG@k (propensity-scored) ficam como extensão futura.

Definição da segmentação (decisão registrada por honestidade): restringimos TANTO o
ranking QUANTO o gold ao conjunto de rótulos do segmento. Assim "tail P@k" mede a
recuperação dos rótulos-gold de cauda usando os candidatos de cauda — direto ao
ponto da RQ e fiel ao desenho 64+64 (cada classe tem sua fatia de candidatos). O
RAG-Fuse restringe só o ranking (índice por classe) e avalia contra o gold inteiro;
documentamos a diferença. Queries sem nenhum gold no segmento são excluídas daquele
recorte (não há o que medir).

Protocolo: avalia por fold (qrels = `label_cols[qid]`) e reporta média ± desvio
entre os folds — "averaged across the five test splits", como o artigo.

Métricas clássicas via `ranx` (fonte da verdade); import lazy (igual a fusion.py).

Uso:
    python -m src.metrics              # compara sparse, dense e o melhor par fundido
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fusion import load_run, run_to_dict
from src.retrieve_sparse import head_tail_split
from src.splits import PooledData, load_pooled

SEGMENTS = ("overall", "head", "tail")


@dataclass
class MetricsConfig:
    raw_dir: str = "data/eurlex4k/raw"
    runs_dir: str = "data/eurlex4k/runs"
    ks: tuple[int, ...] = (1, 5, 10)
    kinds: tuple[str, ...] = ("precision", "ndcg", "recall")  # nomes do ranx
    head_frac: float = 0.20          # Pareto: 20% mais frequentes = cabeça (= sparse/dense)
    n_folds: int = 5
    folds: tuple[int, ...] | None = None   # None = todos; subconjunto p/ "3 dos 5 folds"
    seed: int = 42                   # (não usado aqui; folds vêm dos qids dos runs)

    def fold_ids(self) -> tuple[int, ...]:
        return self.folds if self.folds is not None else tuple(range(self.n_folds))
    # par fundido a avaliar por padrão (melhor do artigo)
    norm: str = "zmuv"
    method: str = "combmnz"

    def fused_template(self) -> str:
        return f"fused.{self.norm}.{self.method}.fold{{fold}}.trec"


# ─────────────────────────── funções puras (testáveis) ─────────────────────────────

def label_to_col(label_id: str) -> int:
    """'label_123' → 123 (chave de rótulo dos runs TREC)."""
    return int(label_id.split("_", 1)[1])


def metric_names(ks: tuple[int, ...], kinds: tuple[str, ...]) -> list[str]:
    """Nomes de métrica do ranx: ('precision',)×(1,5,10) → ['precision@1', ...]."""
    return [f"{kind}@{k}" for kind in kinds for k in ks]


def build_qrels(pooled: PooledData) -> dict[str, dict[str, float]]:
    """Gold por doc global: {qid: {label_{col}: 1.0}} (relevância binária)."""
    return {
        str(i): {f"label_{c}": 1.0 for c in cols}
        for i, cols in enumerate(pooled.label_cols)
    }


def _filter_to_cols(scores: dict[str, float], cols: set[int]) -> dict[str, float]:
    return {l: s for l, s in scores.items() if label_to_col(l) in cols}


def segment(
    run: dict[str, dict[str, float]],
    qrels: dict[str, dict[str, float]],
    cols: set[int],
) -> tuple[dict, dict]:
    """Restringe ranking E gold aos rótulos de `cols`. Queries sem gold no segmento
    são descartadas (não avaliáveis nesse recorte)."""
    run_seg: dict[str, dict[str, float]] = {}
    qrels_seg: dict[str, dict[str, float]] = {}
    for qid, gold in qrels.items():
        g = _filter_to_cols(gold, cols)
        if not g:
            continue
        qrels_seg[qid] = g
        run_seg[qid] = _filter_to_cols(run.get(qid, {}), cols)
    return run_seg, qrels_seg


def evaluate(
    run: dict[str, dict[str, float]],
    qrels: dict[str, dict[str, float]],
    ks: tuple[int, ...],
    kinds: tuple[str, ...],
) -> dict[str, float]:
    """P@k/nDCG@k/Recall@k via ranx (média sobre as queries). Import lazy do ranx."""
    from ranx import Qrels, Run
    from ranx import evaluate as ranx_evaluate

    names = metric_names(ks, kinds)
    if not qrels:
        return {n: float("nan") for n in names}
    # ranx exige inner-dict não-vazio no Run; query sem candidato no segmento vira
    # um placeholder que não casa com o gold (todas as métricas = 0 p/ ela).
    run_safe = {q: (docs if docs else {"__none__": 0.0}) for q, docs in run.items()}
    res = ranx_evaluate(Qrels(qrels), Run(run_safe), names)
    if not isinstance(res, dict):     # ranx devolve float quando há 1 só métrica
        res = {names[0]: res}
    return {n: float(res[n]) for n in names}


def evaluate_segmented(
    run: dict[str, dict[str, float]],
    qrels: dict[str, dict[str, float]],
    head: set[int],
    tail: set[int],
    cfg: MetricsConfig,
) -> dict[str, dict[str, float]]:
    """{overall, head, tail} → cada um um dict de métricas."""
    run_h, qrels_h = segment(run, qrels, head)
    run_t, qrels_t = segment(run, qrels, tail)
    return {
        "overall": evaluate(run, qrels, cfg.ks, cfg.kinds),
        "head": evaluate(run_h, qrels_h, cfg.ks, cfg.kinds),
        "tail": evaluate(run_t, qrels_t, cfg.ks, cfg.kinds),
    }


def aggregate(per_fold: list[dict[str, float]]) -> dict[str, tuple[float, float]]:
    """Lista de dicts de métricas (1 por fold) → {métrica: (média, desvio)}."""
    keys = per_fold[0].keys()
    out: dict[str, tuple[float, float]] = {}
    for key in keys:
        vals = np.array([pf[key] for pf in per_fold], dtype=float)
        if np.all(np.isnan(vals)):     # segmento sem gold em todos os folds (degenerado)
            out[key] = (float("nan"), float("nan"))
        else:
            out[key] = (float(np.nanmean(vals)), float(np.nanstd(vals)))
    return out


# ─────────────────────────── orquestração por run-set ──────────────────────────────

def evaluate_run_set(
    path_template: str,
    cfg: MetricsConfig,
    pooled: PooledData,
    head: set[int],
    tail: set[int],
) -> dict[str, dict[str, tuple[float, float]]]:
    """Avalia um conjunto de runs (1 por fold, ex.: 'sparse.fold{fold}.trec') e
    agrega entre folds. Retorna {segmento: {métrica: (média, desvio)}}.

    Pula folds cujo arquivo não existe (avalia os disponíveis e avisa)."""
    qrels_all = build_qrels(pooled)
    per_fold: dict[str, list[dict[str, float]]] = {s: [] for s in SEGMENTS}
    n_used = 0
    for f in cfg.fold_ids():
        path = os.path.join(cfg.runs_dir, path_template.format(fold=f))
        if not os.path.exists(path):
            print(f"  [aviso] {path} ausente — fold {f} ignorado")
            continue
        run = run_to_dict(load_run(path))
        qrels_fold = {qid: qrels_all[qid] for qid in run if qid in qrels_all}
        seg = evaluate_segmented(run, qrels_fold, head, tail, cfg)
        for s in SEGMENTS:
            per_fold[s].append(seg[s])
        n_used += 1
    if n_used == 0:
        raise FileNotFoundError(f"nenhum run encontrado para o padrão {path_template!r}")
    return {s: aggregate(per_fold[s]) for s in SEGMENTS}


def format_report(label: str, results: dict, cfg: MetricsConfig) -> str:
    """Tabela legível: linhas = segmento, colunas = métrica (média±desvio)."""
    names = metric_names(cfg.ks, cfg.kinds)
    lines = [f"\n### {label}"]
    header = f"{'segmento':<8} | " + " | ".join(f"{n:>14}" for n in names)
    lines.append(header)
    lines.append("-" * len(header))
    for seg in SEGMENTS:
        cells = []
        for n in names:
            mean, std = results[seg][n]
            cells.append(f"{mean:.4f}±{std:.4f}")
        lines.append(f"{seg:<8} | " + " | ".join(f"{c:>14}" for c in cells))
    return "\n".join(lines)


def run_report(cfg: MetricsConfig | None = None) -> None:
    """Compara os recuperadores base (sparse, dense) e o melhor par fundido."""
    cfg = cfg or MetricsConfig()
    pooled = load_pooled(cfg.raw_dir)
    head, tail = head_tail_split(pooled.label_cols, pooled.n_labels, cfg.head_frac)
    print(
        f"avaliação: {len(pooled)} docs | cabeça {len(head)} / cauda {len(tail)} "
        f"| k∈{list(cfg.ks)} | métricas: {list(cfg.kinds)} | folds {list(cfg.fold_ids())} (média±desvio)"
    )

    run_sets = {
        "sparse": "sparse.fold{fold}.trec",
        "dense": "dense.fold{fold}.trec",
        f"fused[{cfg.method}+{cfg.norm}]": cfg.fused_template(),
    }
    for label, template in run_sets.items():
        try:
            results = evaluate_run_set(template, cfg, pooled, head, tail)
            print(format_report(label, results, cfg))
        except FileNotFoundError as e:
            print(f"\n### {label}\n  pulado: {e}")


def main(cfg: MetricsConfig | None = None) -> None:
    import argparse

    from src.data import add_dataset_arg, add_folds_arg, apply_dataset, parse_folds

    parser = argparse.ArgumentParser(description="Avaliação segmentada (overall/cabeça/cauda)")
    add_dataset_arg(parser)
    add_folds_arg(parser)
    args, _ = parser.parse_known_args()

    cfg = cfg or MetricsConfig()
    apply_dataset(cfg, args.dataset)
    cfg.folds = parse_folds(args.folds)
    run_report(cfg)


if __name__ == "__main__":
    main()
