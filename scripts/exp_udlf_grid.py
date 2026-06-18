"""UDLF (CPRR/LHRR/RFE) na MESMA arena da grade ranx — tabela comparativa unificada.

O ponto científico (correção do Daniel): os métodos UDLF têm que aparecer no MESMO
ranking dos 25 métodos ranx × 7 normalizações, não numa comparação isolada. Para ser
justo, todos rodam nas MESMAS condições: mesmos folds e MESMOS inputs (sparse-128 ∪
dense-128, profundidade padrão — NÃO os "blocos fundos", que dariam vantagem só ao UDLF;
a sensibilidade ao L=400 de design fica como análise à parte).

Fluxo (resumável):
  1. roda CPRR/LHRR/RFE no Eurlex, profundidade PADRÃO, nos folds da grade (pula trec
     já existente — ex.: udlf.cprr.fold0.trec do 1º resultado);
  2. agrega cada método sobre os folds (mesma métrica/segmentação do metrics.py);
  3. se data/eurlex4k/results/gridsearch.csv existir (puxado da Brev), MESCLA as linhas
     UDLF na grade e imprime UM ranking único por tail nDCG@5 + grava o CSV unificado.
"""
from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.metrics import MetricsConfig, SEGMENTS, evaluate_run_set, metric_names
from src.retrieve_sparse import head_tail_split
from src.splits import load_pooled
from src.udlf_fusion import UdlfConfig
from src.udlf_fusion import run_cv as udlf_run_cv

METHODS = ["cprr", "lhrr", "rfe"]
KEY_METRIC = ("tail", "ndcg@5")  # ordenação do ranking


def run_udlf_standard(raw: str, runs: str, folds: tuple[int, ...], eval_sample: int) -> None:
    """Gera udlf.<m>.fold<k>.trec na profundidade PADRÃO, pulando os que já existem."""
    for m in METHODS:
        cfg = UdlfConfig(raw_dir=raw, runs_dir=runs, method=m, mode="fusion",
                         folds=folds, eval_sample=eval_sample)  # defaults = 128/L~216
        missing = [k for k in folds if not os.path.exists(cfg.out_path(k))]
        if not missing:
            print(f"[{m}] todos os folds {list(folds)} já existem — pulando")
            continue
        cfg.folds = tuple(missing)
        smp = f", eval_sample={eval_sample}" if eval_sample else ""
        print(f"[{m}] rodando folds {missing} (K/T = default oficial, n_candidates=128{smp})")
        udlf_run_cv(cfg)


def udlf_rows(pooled, head, tail, mcfg) -> list[dict]:
    """Avalia cada método UDLF agregado sobre os folds → linhas no esquema da grade."""
    rows = []
    for m in METHODS:
        try:
            res = evaluate_run_set("udlf." + m + ".fold{fold}.trec", mcfg, pooled, head, tail)
        except FileNotFoundError:
            print(f"[aviso] sem runs de {m} — pulando da tabela")
            continue
        rows.append({"method": f"udlf-{m}", "norm": "—", "agg": res})
    return rows


def load_grid(path: str, names: list[str]) -> list[dict]:
    """Lê o CSV long-format da grade → linhas {method, norm, agg{seg}{metric}=(mean,std)}."""
    cells: dict[tuple, dict] = {}
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            key = (r["method"], r["norm"])
            cells.setdefault(key, {s: {} for s in SEGMENTS})
            cells[key][r["segment"]][r["metric"]] = (float(r["mean"]), float(r["std"]))
    return [{"method": m, "norm": n, "agg": agg} for (m, n), agg in cells.items()]


def key_val(row: dict) -> float:
    seg, met = KEY_METRIC
    return row["agg"].get(seg, {}).get(met, (float("-inf"), 0.0))[0]


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="UDLF (CPRR/LHRR/RFE) na arena da grade ranx")
    ap.add_argument("--dataset", default="eurlex4k", help="ex.: eurlex4k, wiki10-31k, amazoncat-13k")
    ap.add_argument("--folds", default=None, help="ex.: '0,1,2' (default: 0-4)")
    ap.add_argument("--eval-sample", type=int, default=0,
                    help="subamostra N consultas/fold (escala; deve casar com a grade ranx, ex.: 20000 no AmazonCat)")
    ap.add_argument("--grid-csv", default="gridsearch.csv",
                    help="CSV da grade ranx p/ o merge (em results/; ex.: gridsearch_sample.csv no AmazonCat)")
    a = ap.parse_args()
    ds = a.dataset
    raw, runs = f"data/{ds}/raw", f"data/{ds}/runs"
    folds = tuple(int(x) for x in a.folds.split(",")) if a.folds else (0, 1, 2, 3, 4)
    grid_csv = os.path.join("data", ds, "results", a.grid_csv)
    merged_csv = os.path.join("data", ds, "results", "gridsearch_with_udlf.csv")

    run_udlf_standard(raw, runs, folds, a.eval_sample)

    pooled = load_pooled(raw)
    head, tail = head_tail_split(pooled.label_cols, pooled.n_labels, 0.20)
    mcfg = MetricsConfig(raw_dir=raw, runs_dir=runs, ks=(1, 5, 10),
                         kinds=("precision", "ndcg", "recall"), folds=folds)
    names = metric_names(mcfg.ks, mcfg.kinds)

    udlf = udlf_rows(pooled, head, tail, mcfg)
    print("\n=== UDLF (agregado sobre folds, tail nDCG@5) ===")
    for r in sorted(udlf, key=key_val, reverse=True):
        m, s = r["agg"]["tail"]["ndcg@5"]
        print(f"  {r['method']:<12} tail nDCG@5 = {m:.4f} ± {s:.4f}")

    if not os.path.exists(grid_csv):
        print(f"\n[grid] {grid_csv} ainda não está local — puxe da Brev para o merge final.")
        print("       (os runs UDLF já estão gerados; rode este script de novo após puxar)")
        return

    grid = load_grid(grid_csv, names)
    combined = sorted(grid + udlf, key=key_val, reverse=True)

    seg, met = KEY_METRIC
    print(f"\n{'='*72}\n{ds} — ranking ÚNICO por {seg} {met} ({len(combined)} métodos: "
          f"{len(grid)} ranx + {len(udlf)} UDLF)\n{'='*72}")
    print(f"{'#':>3}  {'método+norm':<24} | {'tail n@5':>10} | {'head n@5':>10} | {'over n@5':>10}")
    print("-" * 72)
    for i, r in enumerate(combined, 1):
        tag = f"{r['method']}+{r['norm']}"
        is_udlf = r["method"].startswith("udlf-")
        mark = " ◀ UDLF" if is_udlf else ""
        t = r["agg"]["tail"]["ndcg@5"][0]
        h = r["agg"]["head"]["ndcg@5"][0]
        o = r["agg"]["overall"]["ndcg@5"][0]
        if i <= 10 or is_udlf:
            print(f"{i:>3}  {tag:<24} | {t:>10.4f} | {h:>10.4f} | {o:>10.4f}{mark}")

    # grava CSV unificado (mesmo esquema long da grade)
    with open(merged_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["method", "norm", "segment", "metric", "mean", "std"])
        for r in combined:
            for s in SEGMENTS:
                for n in names:
                    mean, std = r["agg"][s].get(n, (float("nan"), float("nan")))
                    w.writerow([r["method"], r["norm"], s, n, f"{mean:.6f}", f"{std:.6f}"])
    print(f"\nCSV unificado salvo: {merged_csv}")


if __name__ == "__main__":
    main()
