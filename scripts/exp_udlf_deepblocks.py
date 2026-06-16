"""Experimento: CPRR com blocos PROFUNDOS (L ≈ default do método) no Eurlex fold 0.

Motivação (ver docs/udlf-integration.md): o 1º resultado do CPRR (tail nDCG@5 0.279)
rodou com K=20, T=2 (= defaults oficiais do config.ini do UDLF) mas L ≈ 216 — bem
abaixo do default L=400 do CPRR. O L trava no tamanho do bloco, que trava na união
esparso∪denso, que trava na profundidade da recuperação (top-128/lado por design).

Este script dá ao CPRR a profundidade que ele foi desenhado para usar, RE-RECUPERANDO
o ESPARSO mais fundo (BM25/CPU, local) — o denso fica em 128 (sem modelo/GPU local;
re-recuperar o denso fundo é tarefa da Brev). É, portanto, um teste PARCIAL/assimétrico
(esparso fundo + denso raso), honesto sobre essa limitação. Se mesmo assim o CPRR não
se mexer, reforça que o gargalo é a ADAPTAÇÃO (grafo rótulo→rótulo / hack bipartido),
não o L. NÃO sobrescreve nenhum run canônico (escreve em *.deep.fold0.trec).
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fusion import fuse_runs, load_run, run_to_dict, save_run
from src.metrics import (
    MetricsConfig,
    build_qrels,
    evaluate_segmented,
)
from src.retrieve_sparse import SparseConfig, head_tail_split
from src.retrieve_sparse import run_cv as sparse_run_cv
from src.splits import load_pooled
from src.udlf_fusion import UdlfConfig
from src.udlf_fusion import run_cv as udlf_run_cv

RAW = "data/eurlex4k/raw"
RUNS = "data/eurlex4k/runs"
FOLD = 0

# métodos UDLF a avaliar (cada um roda no SEU default oficial de K/T; ver udlf_fusion)
METHODS = ["cprr", "lhrr", "rfe"]

# --- profundidade da re-recuperação esparsa e do bloco UDLF (ajustáveis) -------------
SPARSE_NUM_LABELS = 200   # 200 cabeça + 200 cauda = 400 candidatos esparsos (era 64+64=128)
SPARSE_CUTOFF = 200       # docs-vizinhos de onde os rótulos saem (era 100)
N_CANDIDATES = 400        # top-C por recuperador no bloco UDLF (era 128)
BLOCK_BATCH = 30          # queries por chamada do UDLF (RAM: n=batch×bloco; n²×4 bytes)


def deepen_sparse() -> str:
    """Re-recupera o esparso do fold 0 mais fundo → sparse.deep.fold0.trec."""
    cfg = SparseConfig(
        raw_dir=RAW, runs_dir=RUNS,
        out_template="sparse.deep.fold{fold}.trec",
        num_labels=SPARSE_NUM_LABELS, cutoff=SPARSE_CUTOFF,
        resume=True,
    )
    out = os.path.join(RUNS, cfg.out_template.format(fold=FOLD))
    if os.path.exists(out):
        print(f"[esparso fundo] já existe {out} — pulando")
    else:
        print(f"[esparso fundo] re-recuperando fold {FOLD} (num_labels={SPARSE_NUM_LABELS}, "
              f"cutoff={SPARSE_CUTOFF}) → {out}")
        sparse_run_cv(cfg, only_fold=FOLD)
    return out


def run_method_deep(method: str) -> str:
    """Roda <method> FUSION com esparso fundo + denso(128), blocos grandes → udlf.<m>.deep.fold0."""
    cfg = UdlfConfig(
        raw_dir=RAW, runs_dir=RUNS,
        sparse_template="sparse.deep.fold{fold}.trec",   # esparso FUNDO
        dense_template="dense.fold{fold}.trec",          # denso raso (128) — limitação local
        out_template="udlf.{method}.deep.fold{fold}.trec",
        method=method, mode="fusion",
        n_candidates=N_CANDIDATES, block_batch=BLOCK_BATCH,
        folds=(FOLD,),
    )
    out = cfg.out_path(FOLD)
    if os.path.exists(out):
        print(f"[{method} fundo] já existe {out} — pulando")
        return out
    print(f"[{method} fundo] K/T = default oficial | n_candidates={N_CANDIDATES} "
          f"block_batch={BLOCK_BATCH} → {out}")
    udlf_run_cv(cfg)
    return out


def tail_head_ndcg(run_path: str, pooled, head, tail, mcfg) -> dict:
    """tail/head/overall nDCG@5 de um run (mesma avaliação do metrics.py)."""
    qrels_all = build_qrels(pooled)
    run = run_to_dict(load_run(run_path))
    qrels = {qid: qrels_all[qid] for qid in run if qid in qrels_all}
    seg = evaluate_segmented(run, qrels, head, tail, mcfg)
    return {s: seg[s].get("ndcg@5", float("nan")) for s in ("overall", "head", "tail")}


def main() -> None:
    deepen_sparse()
    deep_outs = {m: run_method_deep(m) for m in METHODS}

    pooled = load_pooled(RAW)
    head, tail = head_tail_split(pooled.label_cols, pooled.n_labels, 0.20)
    mcfg = MetricsConfig(raw_dir=RAW, runs_dir=RUNS, ks=(5,), kinds=("ndcg",))

    # baseline ranx (melhor par do artigo) fundido na hora, p/ referência
    sp_o = load_run(os.path.join(RUNS, f"sparse.fold{FOLD}.trec"))
    de_o = load_run(os.path.join(RUNS, f"dense.fold{FOLD}.trec"))
    best_ranx = os.path.join(RUNS, f"_tmp.combmnz.zmuv.fold{FOLD}.trec")
    save_run(fuse_runs([sp_o, de_o], norm="zmuv", method="combmnz"), best_ranx)

    runs = {
        "esparso (128)":        os.path.join(RUNS, f"sparse.fold{FOLD}.trec"),
        "esparso FUNDO (400)":  os.path.join(RUNS, f"sparse.deep.fold{FOLD}.trec"),
        "denso (128)":          os.path.join(RUNS, f"dense.fold{FOLD}.trec"),
        "combmnz+zmuv (ranx)":  best_ranx,
        "CPRR orig (L~216)":    os.path.join(RUNS, f"udlf.cprr.fold{FOLD}.trec"),
    }
    for m in METHODS:                       # cada método fundo (L~400)
        runs[f"{m.upper()} FUNDO (L~400)"] = deep_outs[m]
    print(f"\n{'='*64}\nEurlex fold {FOLD} — nDCG@5 (cabeça {len(head)} / cauda {len(tail)})\n{'='*64}")
    print(f"{'run':<24} | {'overall':>9} | {'head':>9} | {'tail':>9}")
    print("-" * 64)
    for label, path in runs.items():
        if not os.path.exists(path):
            print(f"{label:<24} | {'AUSENTE':>9}")
            continue
        r = tail_head_ndcg(path, pooled, head, tail, mcfg)
        print(f"{label:<24} | {r['overall']:>9.4f} | {r['head']:>9.4f} | {r['tail']:>9.4f}")
    os.remove(best_ranx)


if __name__ == "__main__":
    main()
