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
import io
import os
import random
import sys
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field

# numba com fork-safe threading layer: a paralelização do grid usa um process pool
# por fork; o "workqueue" evita o deadlock dos layers OpenMP/TBB sob fork. Setado
# ANTES de qualquer import de ranx/numba (que é lazy). Inofensivo no modo serial.
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

if __package__ in {None, ""}:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fusion import (
    METHODS,
    NORMS,
    SUPERVISED,
    fuse_runs,
    learn_fusion_params,
    load_run,
    method_params,
    run_to_dict,
)
from src.metrics import (
    SEGMENTS,
    MetricsConfig,
    aggregate,
    build_qrels,
    evaluate_segmented,
    label_to_col,
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
    folds: tuple[int, ...] | None = None   # None = todos; subconjunto p/ "3 dos 5 folds"
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
    workers: int = 1   # >1 → avalia as combos (norma × fusão) em paralelo (process pool, fork)
    resume: bool = True   # retoma do checkpoint (.ckpt.csv) pulando combos já feitas
    opt_sample: int = 0   # >0 → subamostra o treino da fusão supervisionada p/ N queries (acelera optimize)
    opt_repeats: int = 1  # >1 → K amostras independentes + seleção do melhor no treino cheio (robustez)
    eval_sample: int = 0  # >0 → subamostra as queries de CADA fold p/ N (fusão+avaliação+treino); escala (AmazonCat/670K)
    skip_methods: tuple[str, ...] = ()  # fusões a EXCLUIR da grade (ex.: caras no fold cheio: condorcet/probfuse/...)

    def fold_ids(self) -> tuple[int, ...]:
        return self.folds if self.folds is not None else tuple(range(self.n_folds))

    def metrics_config(self) -> MetricsConfig:
        return MetricsConfig(ks=self.ks, kinds=self.kinds, head_frac=self.head_frac,
                             n_folds=self.n_folds, folds=self.folds)


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


def _restrict_qrels(qrels_fold: dict, cols: set[int]) -> dict:
    """Restringe os qrels aos rótulos de `cols` (ex.: cauda); descarta query sem gold
    no recorte — é o sinal que faz o optimize_fusion mirar a métrica daquele segmento."""
    out: dict[str, dict[str, float]] = {}
    for qid, gold in qrels_fold.items():
        g = {l: r for l, r in gold.items() if label_to_col(l) in cols}
        if g:
            out[qid] = g
    return out


def _merge_runs(runs):
    """Une vários ranx.Run de qids disjuntos (folds de treino) num só Run."""
    from ranx import Run

    merged: dict[str, dict[str, float]] = {}
    for r in runs:
        merged.update(r.to_dict())
    return Run(merged)


def evaluate_combo_supervised(
    norm: str,
    method: str,
    fold_runs: list[tuple],
    head: set[int],
    tail: set[int],
    mcfg: MetricsConfig,
    select_segment: str,
    select_metric: str,
    opt_sample: int = 0,
    opt_repeats: int = 1,
) -> dict[str, dict[str, tuple[float, float]]]:
    """Avalia uma fusão SUPERVISIONADA via CV ANINHADA: para fundir o fold k, aprende
    os parâmetros nos OUTROS folds (params disjuntos do teste → sem vazamento), com os
    qrels restritos ao segmento de seleção (cauda) para otimizar a métrica de cauda.
    Aplica os params aprendidos ao fold k e avalia segmentado como qualquer método.

    Requer ≥2 folds (1 p/ treino, 1 p/ teste); com <2 devolve NaN (sem o que treinar)."""
    seg_cols = tail if select_segment == "tail" else (head if select_segment == "head" else None)
    per_fold: dict[str, list[dict]] = {s: [] for s in SEGMENTS}
    n = len(fold_runs)
    for k in range(n):
        train_idx = [j for j in range(n) if j != k]
        if not train_idx:                       # 1 fold só: nada para treinar
            for s in SEGMENTS:
                per_fold[s].append({m: float("nan") for m in metric_names(mcfg.ks, mcfg.kinds)})
            continue
        train_sparse = _merge_runs([fold_runs[j][0] for j in train_idx]); train_sparse.name = "sparse"
        train_dense = _merge_runs([fold_runs[j][1] for j in train_idx]); train_dense.name = "dense"
        train_qrels: dict[str, dict[str, float]] = {}
        for j in train_idx:
            q = _restrict_qrels(fold_runs[j][2], seg_cols) if seg_cols else fold_runs[j][2]
            train_qrels.update(q)
        params = learn_fusion_params(
            train_qrels, [train_sparse, train_dense], norm, method,
            metric=select_metric, sample_size=opt_sample, repeats=opt_repeats,
        )
        sparse_k, dense_k, qrels_k = fold_runs[k]
        fused = fuse_runs([sparse_k, dense_k], norm=norm, method=method, params=params)
        seg = evaluate_segmented(run_to_dict(fused), qrels_k, head, tail, mcfg)
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
    for f in cfg.fold_ids():
        sp = os.path.join(cfg.runs_dir, cfg.sparse_template.format(fold=f))
        de = os.path.join(cfg.runs_dir, cfg.dense_template.format(fold=f))
        for p in (sp, de):
            if not os.path.exists(p):
                raise FileNotFoundError(f"run base ausente: {p} (rode retrieve_sparse/retrieve_dense)")
        sparse_run = load_run(sp, name="sparse")
        dense_run = load_run(de, name="dense")
        fold_runs.append(subsample_fold(sparse_run, dense_run, qrels_all, cfg.eval_sample))
    return fold_runs, head, tail


def subsample_fold(sparse_run, dense_run, qrels_all: dict, n: int, seed: int = 42) -> tuple:
    """Restringe um fold (sparse_run, dense_run, qrels) a uma subamostra DETERMINÍSTICA de
    `n` queries (seed fixa). Com n=0 (ou fold menor que n) devolve o fold inteiro.

    Aplicada 1× ao carregar → a MESMA amostra vale p/ fusão, avaliação E treino
    supervisionado de TODOS os métodos. Como a métrica é a média por query, a amostra
    aleatória é um estimador não-enviesado (mesma amostra p/ todos → ranking justo);
    corta o custo O(n²) do condorcet e a RAM da fusão+avaliação em folds gigantes."""
    sd = run_to_dict(sparse_run)
    if not n or len(sd) <= n:
        qids = list(sd.keys())
        return sparse_run, dense_run, {q: qrels_all[q] for q in qids if q in qrels_all}
    from ranx import Run
    dd = run_to_dict(dense_run)
    sample = random.Random(seed).sample(sorted(sd), n)
    sparse_run = Run({q: sd[q] for q in sample}, name="sparse")
    dense_run = Run({q: dd[q] for q in sample if q in dd}, name="dense")
    qrels_fold = {q: qrels_all[q] for q in sample if q in qrels_all}
    return sparse_run, dense_run, qrels_fold


def _evaluate_cell(norm: str, method: str, fold_runs, head, tail, mcfg, cfg) -> dict:
    """Avalia UMA combinação (norma × fusão) — supervisionada (CV aninhada) ou não.
    Unidade de trabalho compartilhada pelos caminhos serial e paralelo."""
    if method in SUPERVISED:
        agg = evaluate_combo_supervised(
            norm, method, fold_runs, head, tail, mcfg,
            cfg.select_segment, cfg.select_metric,
            opt_sample=cfg.opt_sample, opt_repeats=cfg.opt_repeats,
        )
    else:
        params = method_params(method, k=cfg.rrf_k, phi=cfg.rbc_phi, gamma=cfg.gmnz_gamma)
        agg = evaluate_combo(norm, method, fold_runs, head, tail, mcfg, params)
    return {"norm": norm, "method": method, "agg": agg}


# Estado compartilhado por FORK com os workers do pool — evita picklar os runs base
# (ranx.Run). Setado no pai ANTES de criar o pool; os filhos herdam por copy-on-write.
_SHARED: dict = {}


def _pool_init() -> None:
    """Init de cada worker: numba single-thread. Evita oversubscrição (N workers × M
    threads cada) e mantém a paralelização 'educada' no host compartilhado."""
    try:
        import numba
        numba.set_num_threads(1)
    except Exception:
        pass


def _pool_eval(task):
    method, norm = task        # all_cells = (method, norm)
    s = _SHARED
    return _evaluate_cell(norm, method, s["fold_runs"], s["head"], s["tail"], s["mcfg"], s["cfg"])


# ─────────────────────── checkpoint (resume + à prova de crash) ─────────────────────
# Cada combo concluída é gravada NA HORA num checkpoint long-format (.ckpt.csv). Se o
# pool quebrar (ex.: worker morto por OOM), o que já terminou está salvo; relançar (ou
# o auto-retry abaixo) retoma só as combos que faltam. Nada de "perdeu tudo".

def checkpoint_path(out_csv: str) -> str:
    base, ext = os.path.splitext(out_csv)
    return f"{base}.ckpt{ext or '.csv'}"


def _cell_rows(rec: dict, names: list[str]) -> list[list[str]]:
    return [[rec["method"], rec["norm"], seg, n,
             f"{rec['agg'][seg][n][0]:.6f}", f"{rec['agg'][seg][n][1]:.6f}"]
            for seg in SEGMENTS for n in names]


def _append_checkpoint(path: str, rec: dict, names: list[str]) -> None:
    """Grava as linhas de UMA combo em escrita única (atômica o suficiente) + fsync,
    pra um crash não deixar a combo pela metade."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    buf = io.StringIO()
    csv.writer(buf).writerows(_cell_rows(rec, names))
    with open(path, "a", newline="", encoding="utf-8") as fh:
        fh.write(buf.getvalue())
        fh.flush()
        os.fsync(fh.fileno())


def load_checkpoint(path: str, names: list[str]) -> dict:
    """{(method,norm): agg} só para combos COMPLETAS (todas as métricas presentes).
    Tolera duplicatas (último valor vence) — relançar nunca corrompe."""
    if not os.path.exists(path):
        return {}
    cells: dict = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if len(row) != 6:
                continue
            m, n, seg, metric, mean, std = row
            cells.setdefault((m, n), {}).setdefault(seg, {})[metric] = (float(mean), float(std))
    expected = len(SEGMENTS) * len(names)
    return {k: agg for k, agg in cells.items()
            if sum(len(v) for v in agg.values()) >= expected}


def _run_cells(cells, fold_runs, head, tail, mcfg, cfg, workers, ckpt, names, base_done, total):
    """Processa uma leva de combos, gravando cada uma no checkpoint ao concluir.
    workers>1 → process pool (fork); pode levantar BrokenProcessPool (tratado acima)."""
    done = base_done
    if workers > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed

        _SHARED.update(fold_runs=fold_runs, head=head, tail=tail, mcfg=mcfg, cfg=cfg)
        ctx = mp.get_context("fork")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx,
                                 initializer=_pool_init) as ex:
            futs = {ex.submit(_pool_eval, c): c for c in cells}
            for fut in as_completed(futs):
                rec = fut.result()      # pode levantar BrokenProcessPool (worker morto)
                _append_checkpoint(ckpt, rec, names)
                done += 1
                print(f"  ({done}/{total}) {rec['method']}+{rec['norm']}", flush=True)
    else:
        for (method, norm) in cells:
            rec = _evaluate_cell(norm, method, fold_runs, head, tail, mcfg, cfg)
            _append_checkpoint(ckpt, rec, names)
            done += 1
            print(f"  ({done}/{total}) {method}+{norm}", flush=True)


def run_grid(cfg: GridConfig | None = None) -> list[dict]:
    """Avalia todas as combinações (norms × methods) e retorna os registros
    [{norm, method, agg}], ordenados pela métrica de seleção.

    À PROVA DE CRASH: cada combo vai pro checkpoint ao concluir; se o pool quebrar
    (OOM de worker), o que terminou está salvo e o run **auto-retoma as restantes
    com metade dos workers** (até serial, se preciso). `cfg.resume` (default) também
    permite continuar de um run interrompido — só relançar o mesmo comando."""
    cfg = cfg or GridConfig()
    mcfg = cfg.metrics_config()
    fold_runs, head, tail = load_fold_runs(cfg)
    names = metric_names(cfg.ks, cfg.kinds)
    all_cells = [(method, norm) for method in cfg.methods for norm in cfg.norms]
    total = len(all_cells)
    ckpt = checkpoint_path(cfg.out_csv)

    if not cfg.resume and os.path.exists(ckpt):
        os.remove(ckpt)                       # --no-resume: começa do zero
    done = load_checkpoint(ckpt, names)
    remaining = [c for c in all_cells if c not in done]

    print(
        f"grid: {len(cfg.norms)} norm × {len(cfg.methods)} fusão = {total} combos "
        f"| folds {list(cfg.fold_ids())} | ranqueando por {cfg.select_segment} {cfg.select_metric}"
        + (f" | {cfg.workers} workers" if cfg.workers and cfg.workers > 1 else "")
        + (f" | resume: {len(done)}/{total} já feitas" if done else "")
    )

    workers = cfg.workers if (cfg.workers and cfg.workers > 1) else 1
    while remaining:
        try:
            _run_cells(remaining, fold_runs, head, tail, mcfg, cfg, workers, ckpt, names,
                       base_done=len(done), total=total)
        except BrokenProcessPool:
            done = load_checkpoint(ckpt, names)          # o que sobreviveu está salvo
            remaining = [c for c in all_cells if c not in done]
            if workers > 1:
                workers = max(1, workers // 2)
                print(f"  [recuperação] pool quebrou (provável OOM) → retomando "
                      f"{len(remaining)} combos com {workers} worker(s)", flush=True)
                continue
            raise                                         # serial também quebrou: erro real
        done = load_checkpoint(ckpt, names)
        remaining = [c for c in all_cells if c not in done]

    records = [{"method": m, "norm": n, "agg": done[(m, n)]} for (m, n) in all_cells]
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
    # final gravado com sucesso → o checkpoint cumpriu o papel, pode sair
    ckpt = checkpoint_path(cfg.out_csv)
    if os.path.exists(ckpt):
        os.remove(ckpt)
    print(f"\nCSV salvo: {cfg.out_csv} ({len(ranked)} combos × {len(SEGMENTS)} segmentos × {len(names)} métricas)")


def main(cfg: GridConfig | None = None) -> None:
    import argparse

    from src.data import add_dataset_arg, add_folds_arg, apply_dataset, parse_folds

    parser = argparse.ArgumentParser(description="Grid search da fusão (norm × método)")
    add_dataset_arg(parser)
    add_folds_arg(parser)
    parser.add_argument("--select", type=str, default=None,
                        help="métrica de seleção no formato segmento:metrica (ex.: tail:ndcg@5)")
    parser.add_argument("--paper", action="store_true",
                        help="só as 6×10 combinações do artigo (exclui rrf/rbc/gmnz/logn_isr/min-max-inverted)")
    parser.add_argument("--top", type=int, default=None, help="quantas combos imprimir")
    parser.add_argument("--workers", type=int, default=None,
                        help="nº de processos paralelos p/ as combos (default 1 = serial). "
                             "Ex.: 16 no host compartilhado da Brev (255 cores)")
    parser.add_argument("--no-resume", action="store_true",
                        help="ignora o checkpoint e refaz todas as combos do zero")
    parser.add_argument("--opt-sample", type=int, default=None,
                        help="subamostra o treino da fusão supervisionada p/ N queries "
                             "(acelera o optimize em datasets grandes; ex.: 2000). "
                             "Não afeta a avaliação, que usa os folds de teste completos")
    parser.add_argument("--opt-repeats", type=int, default=None,
                        help="K amostras independentes + seleção do melhor candidato no "
                             "treino completo (robustez à variância de amostra; ex.: 3). "
                             "Mantenha K×opt-sample << treino p/ preservar a aceleração")
    parser.add_argument("--eval-sample", type=int, default=None,
                        help="subamostra as queries de CADA fold p/ N (fusão+avaliação+treino) "
                             "— escala datasets grandes (AmazonCat/670K) e corta o O(n²) do "
                             "condorcet e a RAM. Estimador não-enviesado; reavalie o topo no "
                             "fold cheio p/ os números finais. Use com --no-resume (muda a base)")
    parser.add_argument("--skip-methods", type=str, default=None,
                        help="lista separada por vírgula de fusões a EXCLUIR da grade (ex.: "
                             "'condorcet,wcondorcet,probfuse,segfuse,slidefuse'). Use p/ reavaliar "
                             "só os métodos baratos no fold cheio sem o O(n²)/OOM dos caros")
    args, _ = parser.parse_known_args()

    cfg = cfg or GridConfig()
    apply_dataset(cfg, args.dataset)
    cfg.folds = parse_folds(args.folds)
    if args.paper:
        cfg.norms, cfg.methods = PAPER_NORMS, PAPER_METHODS
    if args.select:
        seg, met = args.select.split(":", 1)
        cfg.select_segment, cfg.select_metric = seg, met
    if args.top:
        cfg.top_n = args.top
    if args.workers:
        cfg.workers = args.workers
    if args.no_resume:
        cfg.resume = False
    if args.opt_sample:
        cfg.opt_sample = args.opt_sample
    if args.opt_repeats:
        cfg.opt_repeats = args.opt_repeats
    if args.eval_sample:
        cfg.eval_sample = args.eval_sample
    if args.skip_methods:
        skip = {m.strip() for m in args.skip_methods.split(",") if m.strip()}
        unknown = skip - set(cfg.methods)
        if unknown:
            print(f"[aviso] --skip-methods ignora nomes fora da grade: {sorted(unknown)}")
        cfg.methods = tuple(m for m in cfg.methods if m not in skip)
        print(f"[skip] excluídas {sorted(skip & set(METHODS))} → grade com {len(cfg.methods)} fusões")

    ranked = run_grid(cfg)
    print(format_grid_report(ranked, cfg))
    save_csv(ranked, cfg)


if __name__ == "__main__":
    main()
