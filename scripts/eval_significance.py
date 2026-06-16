"""Baselines isolados + significância estatística (Eurlex/Wiki10 — runs locais).

Responde duas lacunas pra fechar a parte experimental:
  (3) BASELINES ISOLADOS — avalia o esparso e o denso SOZINHOS (sem fusão), segmentados
      cabeça/cauda, agregados sobre os folds → quantifica o ganho MARGINAL da fusão.
  (1) SIGNIFICÂNCIA — teste pareado por-query (Wilcoxon signed-rank) + IC bootstrap da
      diferença média, no tail nDCG@5, para as comparações que sustentam as afirmações:
        • topo (melhor par por tail nDCG@5) vs baseline CombMNZ+ZMUV
        • baseline vs denso isolado   (a fusão clássica supera o melhor recuperador só?)
        • topo vs denso isolado       (o melhor par supera o denso?)

Reproduz os runs fundidos pelo MESMO caminho do gridsearch (CV aninhada p/ supervisionados)
→ a média por-query deve bater com o gridsearch.csv (conferimos e avisamos se divergir).
Determinístico (bootstrap com seed fixa). Read-only sobre os runs; não grava nada.
"""
from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src.data import apply_dataset
from src.fusion import SUPERVISED, fuse_runs, method_params, run_to_dict
from src.gridsearch import (
    GridConfig,
    _merge_runs,
    _restrict_qrels,
    load_fold_runs,
)
from src.fusion import learn_fusion_params
from src.metrics import MetricsConfig, evaluate_run_set, segment
from src.splits import load_pooled
from src.retrieve_sparse import head_tail_split

DATASETS = ["eurlex4k", "wiki10-31k"]   # runs locais; AmazonCat/670K = Brev (runs gigantes)
BASE = ("combmnz", "zmuv")
N_BOOT = 2000
SEED = 42


def ndcg5(ranked_labels: list[str], gold: set[str]) -> float:
    dcg = sum(1.0 / math.log2(i + 2) for i, l in enumerate(ranked_labels[:5]) if l in gold)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), 5)))
    return dcg / idcg if idcg > 0 else 0.0


def per_query_tail_ndcg(run_dict: dict, qrels_fold: dict, tail: set[int]) -> dict[str, float]:
    """tail nDCG@5 por query (mesma segmentação do metrics.py: restringe run E gold à cauda;
    queries sem gold de cauda são excluídas)."""
    run_t, qrels_t = segment(run_dict, qrels_fold, tail)
    out = {}
    for qid, gold in qrels_t.items():
        ranked = sorted(run_t.get(qid, {}), key=lambda l: -run_t[qid][l])
        out[qid] = ndcg5(ranked, set(gold))
    return out


def fused_per_fold(method: str, norm: str, fold_runs, head, tail) -> list[tuple[dict, dict]]:
    """Para cada fold k: (run_fundido_dict, qrels_k). Supervisionado → CV aninhada
    (aprende nos outros folds, qrels restritos à cauda), igual ao gridsearch."""
    out = []
    n = len(fold_runs)
    for k in range(n):
        sparse_k, dense_k, qrels_k = fold_runs[k]
        if method in SUPERVISED:
            tr = [j for j in range(n) if j != k]
            ts = _merge_runs([fold_runs[j][0] for j in tr]); ts.name = "sparse"
            td = _merge_runs([fold_runs[j][1] for j in tr]); td.name = "dense"
            tq = {}
            for j in tr:
                tq.update(_restrict_qrels(fold_runs[j][2], tail))
            params = learn_fusion_params(tq, [ts, td], norm, method, metric="ndcg@5")
        else:
            params = method_params(method)
        fused = fuse_runs([sparse_k, dense_k], norm=norm, method=method, params=params)
        out.append((run_to_dict(fused), qrels_k))
    return out


def isolated_per_fold(which: int, fold_runs) -> list[tuple[dict, dict]]:
    """which=0 esparso, 1 denso — run isolado por fold (sem fusão)."""
    return [(run_to_dict(fr[which]), fr[2]) for fr in fold_runs]


def collect(system_folds, tail) -> dict[str, float]:
    """Une as métricas por-query de todos os folds (qids disjuntos)."""
    allq = {}
    for run_dict, qrels_k in system_folds:
        allq.update(per_query_tail_ndcg(run_dict, qrels_k, tail))
    return allq


def paired_test(a: dict, b: dict, label: str) -> dict:
    from scipy.stats import wilcoxon

    qs = sorted(set(a) & set(b))
    da = np.array([a[q] for q in qs]); db = np.array([b[q] for q in qs])
    diff = da - db
    mean_d = float(diff.mean())
    n = len(diff)
    rng = np.random.default_rng(SEED)                       # bootstrap vetorizado (determinístico)
    idx = rng.integers(0, n, size=(N_BOOT, n), dtype=np.int32)
    boots = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    try:
        stat, p = wilcoxon(da, db, alternative="two-sided", zero_method="wilcox")
        p = float(p)
    except ValueError:
        p = float("nan")
    win = int((diff > 0).sum()); lose = int((diff < 0).sum()); tie = int((diff == 0).sum())
    sig = "✓ p<0.05" if (p == p and p < 0.05) else ("× n.s." if p == p else "× (degenerado)")
    print(f"  {label:<34} Δmean={mean_d:+.4f}  IC95%=[{lo:+.4f},{hi:+.4f}]  "
          f"p={p:.2e} {sig}  (W/L/T={win}/{lose}/{tie}, n={n})")
    return {"comparison": label, "delta": mean_d, "ci_lo": float(lo), "ci_hi": float(hi),
            "p": p, "n": n, "win": win, "lose": lose, "tie": tie}


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Baselines isolados + significância (cauda)")
    ap.add_argument("--psp", action="store_true",
                    help="inclui PSP@k/PSnDCG@k nos baselines (exige xclib; presente na Brev)")
    ap.add_argument("--datasets", type=str, default=None,
                    help="lista separada por vírgula (default: eurlex4k,wiki10-31k)")
    cli, _ = ap.parse_known_args()
    datasets = cli.datasets.split(",") if cli.datasets else DATASETS
    base_kinds = ("precision", "ndcg", "recall") + (("psp", "psndcg") if cli.psp else ())
    for ds in datasets:
        print("=" * 78); print(f"# {ds}")
        cfg = GridConfig(); apply_dataset(cfg, ds)
        try:
            fold_runs, head, tail, _inv_psp, _n_labels = load_fold_runs(cfg)
        except FileNotFoundError as e:
            print(f"  runs ausentes: {e}"); continue
        pooled = load_pooled(cfg.raw_dir)
        mcfg = MetricsConfig(raw_dir=cfg.raw_dir, runs_dir=cfg.runs_dir,
                             ks=(1, 5, 10), kinds=base_kinds)  # = schema do grid (+psp se --psp)
        from src.metrics import SEGMENTS, metric_names
        names = metric_names(mcfg.ks, mcfg.kinds)
        resdir = cfg.runs_dir.replace("runs", "results")
        os.makedirs(resdir, exist_ok=True)

        # (3) BASELINES ISOLADOS (agregado sobre folds, segmentado) → baselines.csv (schema do grid)
        print("\n  --- (3) Baselines isolados (mean±std entre folds) ---")
        import csv as _csv
        with open(os.path.join(resdir, "baselines.csv"), "w", newline="") as fh:
            w = _csv.writer(fh); w.writerow(["method", "norm", "segment", "metric", "mean", "std"])
            for label, tmpl in [("sparse", "sparse.fold{fold}.trec"), ("dense", "dense.fold{fold}.trec")]:
                r = evaluate_run_set(tmpl, mcfg, pooled, head, tail)
                for s in SEGMENTS:
                    for nm in names:
                        mean, std = r[s][nm]
                        w.writerow([label, "none", s, nm, f"{mean:.6f}", f"{std:.6f}"])
                tn = r["tail"]["ndcg@5"]; hn = r["head"]["ndcg@5"]
                print(f"    {label:<7} tail nDCG@5={tn[0]:.4f}±{tn[1]:.4f} | head nDCG@5={hn[0]:.4f}±{hn[1]:.4f}")

        # descobre o topo por tail ndcg@5 a partir do CSV
        import csv
        csvp = os.path.join(cfg.runs_dir.replace("runs", "results"), "gridsearch.csv")
        top = None
        if os.path.exists(csvp):
            best = {}
            for row in csv.DictReader(open(csvp)):
                if row["segment"] == "tail" and row["metric"] == "ndcg@5":
                    best[(row["method"], row["norm"])] = float(row["mean"])
            top = max(best, key=best.get)
        print(f"\n  topo por tail nDCG@5 (do CSV): {top}")

        # sistemas para o teste pareado
        sys_top = collect(fused_per_fold(top[0], top[1], fold_runs, head, tail), tail) if top else {}
        sys_base = collect(fused_per_fold(BASE[0], BASE[1], fold_runs, head, tail), tail)
        sys_dense = collect(isolated_per_fold(1, fold_runs), tail)
        sys_sparse = collect(isolated_per_fold(0, fold_runs), tail)

        # sanidade: média por-query bate com o CSV?
        def mean_of(d): return sum(d.values()) / len(d) if d else float("nan")
        print(f"  [sanidade] média por-query: topo={mean_of(sys_top):.4f} base={mean_of(sys_base):.4f} "
              f"dense={mean_of(sys_dense):.4f} sparse={mean_of(sys_sparse):.4f}  (comparar c/ CSV)")

        print("\n  --- (1) Significância no tail nDCG@5 (Wilcoxon pareado + IC95% bootstrap) ---")
        sigrows = []
        if top and top != BASE:
            sigrows.append(paired_test(sys_top, sys_base, f"{top[0]}+{top[1]} vs combmnz+zmuv"))
        sigrows.append(paired_test(sys_base, sys_dense, "combmnz+zmuv vs dense (fusion gain)"))
        if top:
            sigrows.append(paired_test(sys_top, sys_dense, f"{top[0]}+{top[1]} vs dense"))
        sigrows.append(paired_test(sys_base, sys_sparse, "combmnz+zmuv vs sparse"))
        with open(os.path.join(resdir, "significance.csv"), "w", newline="") as fh:
            w = _csv.writer(fh)
            w.writerow(["comparison", "segment", "metric", "delta", "ci_lo", "ci_hi", "p", "n", "win", "lose", "tie"])
            for r in sigrows:
                w.writerow([r["comparison"], "tail", "ndcg@5", f"{r['delta']:.6f}",
                            f"{r['ci_lo']:.6f}", f"{r['ci_hi']:.6f}", f"{r['p']:.3e}",
                            r["n"], r["win"], r["lose"], r["tie"]])
        print(f"  → escrito {resdir}/baselines.csv + significance.csv")
        print()


if __name__ == "__main__":
    main()
