"""Fusão/re-ranking CONTEXTUAL via UDLF (CPRR/LHRR/RFE) — trilha distinta da grade
ranx, avaliada lado a lado (mesmos folds, mesmas métricas cabeça/cauda).

Os métodos do UDLF (Pedronette/UNICAMP-UNESP) exploram a estrutura CONTEXTUAL das
listas (grafos de kNN recíproco, hipergrafos, embeddings de fluxo) para reordenar sem
supervisão — família diferente dos fundidores escore-a-escore do ranx. Ver o desenho
completo e as referências em docs/udlf-integration.md.

## Descompasso e adaptação (bipartido → quadrado)
O UDLF supõe listas item→item (cenário "quadrado": cada elemento é query e item). O
nosso é bipartido (query = documento, item = rótulo). Adaptação **por blocos por
query** (validada no smoke test: blocos disjuntos não vazam):

  - Para cada query q do fold, o bloco B(q) = [q] + candidatos (união dos top-C do
    esparso e do denso).
  - Listas do bloco (formato RK/NUM, índices LOCAIS ao bloco):
      * linha de q     = candidatos ranqueados pelo escore do recuperador (doc→rótulo);
      * linha de cada rótulo = demais rótulos do bloco por similaridade rótulo→rótulo
        (Opção A: co-ocorrência em Y_train do fold — offline, fold-safe).
  - Vários blocos DISJUNTOS são empacotados numa execução (block-diagonal): cada
    bloco recebe uma faixa de índices própria; rótulos compartilhados entre queries
    são DUPLICADOS com índices distintos (universos independentes), então não há
    vazamento. Lote de `block_batch` queries por chamada do pyUDLF.
  - Da saída, extrai-se a linha de cada q → run TREC (escore sintético 1/(rank+1); as
    métricas só usam a ordem) → consumido por metrics.py como qualquer run.

## Modos
  - FUSION(esparso, denso): 2 conjuntos de listas → linha nova na tabela de fusão.
  - UDL: re-ranking de UM run já fundido (ex.: o melhor par do ranx).

Honestidade (ver docs): a adaptação bipartida + o rótulo→rótulo derivado são NOSSOS
(o framework é de CBIR); descritos explicitamente. A saída do UDLF não tem escore —
só a ordem é comparável.

Import do pyUDLF é LAZY (só ao executar) — funções puras (blocos, RK, co-ocorrência,
parsing) são testáveis sem o binário.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp

if __package__ in {None, ""}:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.fusion import load_run, run_to_dict
from src.retrieve_sparse import write_trec
from src.splits import load_pooled, make_folds

# métodos do UDLF sugeridos pelo grupo (nome nosso → nome no UDLF)
UDLF_METHODS: dict[str, str] = {"cprr": "CPRR", "lhrr": "LHRR", "rfe": "RFE"}

# defaults OFICIAIS por método, lidos do config.ini do binário UDLF (~/.pyudlf/bin).
# Cada método tem o SEU K/T de design — não dá para aplicar o do CPRR a todos (LHRR
# usa K=18, não 20). Usados quando não há override, para cada método rodar autêntico.
UDLF_DEFAULT_K: dict[str, int] = {"cprr": 20, "lhrr": 18, "rfe": 20}
UDLF_DEFAULT_T: dict[str, int] = {"cprr": 2, "lhrr": 2, "rfe": 2}


@dataclass
class UdlfConfig:
    raw_dir: str = "data/eurlex4k/raw"
    runs_dir: str = "data/eurlex4k/runs"
    sparse_template: str = "sparse.fold{fold}.trec"
    dense_template: str = "dense.fold{fold}.trec"
    out_template: str = "udlf.{method}.fold{fold}.trec"
    method: str = "cprr"                 # chave de UDLF_METHODS
    mode: str = "fusion"                 # "fusion" (esparso+denso) | "udl" (re-rank de 1 run)
    udl_run_template: str = "fused.zmuv.combmnz.fold{fold}.trec"  # run a re-ranquear no modo udl
    n_candidates: int = 128              # top-C por recuperador → candidatos do bloco
    label_topl: int = 0                  # L do método (0 = tamanho do bloco)
    block_batch: int = 80                # queries por chamada do pyUDLF (n ≈ batch × bloco)
    k_override: int = 0                  # K do kNN; 0 = default oficial do método (CPRR/RFE 20, LHRR 18)
    t_override: int = 0                  # T (iterações); 0 = default oficial do método (todos 2)
    n_folds: int = 5
    folds: tuple[int, ...] | None = None
    seed: int = 42
    binary_path: str | None = None       # caminho do binário UDLF (None = pyUDLF baixa)
    tag: str = "udlf"

    def fold_ids(self) -> tuple[int, ...]:
        return self.folds if self.folds is not None else tuple(range(self.n_folds))

    def out_path(self, fold_id: int) -> str:
        return os.path.join(self.runs_dir, self.out_template.format(
            method=self.method, fold=fold_id))


# ─────────────────────── Opção A: rótulo→rótulo por co-ocorrência ───────────────────

def build_train_label_matrix(train_idx, label_cols: list[list[int]], n_labels: int) -> sp.csc_matrix:
    """Matriz binária docs_treino × rótulos (CSC p/ fatiar colunas rápido). Só docs do
    fold de TREINO → fold-safe (sem vazamento do teste na similaridade rótulo→rótulo)."""
    rows, cols = [], []
    for r, doc in enumerate(train_idx):
        for c in label_cols[doc]:
            rows.append(r)
            cols.append(c)
    data = np.ones(len(rows), dtype=np.float32)
    Y = sp.csr_matrix((data, (rows, cols)), shape=(len(train_idx), n_labels))
    return Y.tocsc()


def block_label_neighbors(block_labels: list[int], Ytr: sp.csc_matrix) -> dict[int, list[int]]:
    """Para cada rótulo do bloco, ordena os DEMAIS rótulos do bloco por similaridade de
    co-ocorrência (cosseno entre colunas de Y_train), restrito ao bloco. Opção A.

    cosseno(a,b) = |docs com a e b| / sqrt(|a|·|b|). Calculado só no bloco (m×m) → barato.
    """
    sub = Ytr[:, block_labels]                       # docs × m (m = tamanho do bloco)
    gram = (sub.T @ sub).toarray()                   # m × m co-ocorrências
    norms = np.sqrt(np.clip(np.diag(gram), 1e-9, None))
    sim = gram / np.outer(norms, norms)              # cosseno
    np.fill_diagonal(sim, -np.inf)                   # self vai pro fim; excluído abaixo
    order = np.argsort(-sim, axis=1)                 # vizinhos por similaridade desc
    return {block_labels[i]: [block_labels[j] for j in order[i] if j != i]
            for i in range(len(block_labels))}       # SEM o próprio rótulo (evita índice duplicado)


# ─────────────────────────── construção de blocos + RK ──────────────────────────────

def block_candidates(sparse_scores: dict[str, float], dense_scores: dict[str, float],
                     n_candidates: int) -> list[int]:
    """Candidatos do bloco = união dos top-`n_candidates` rótulos do esparso e do denso,
    em ordem determinística (por coluna do rótulo). label_id 'label_C' → C."""
    def topc(scores):
        items = sorted(scores.items(), key=lambda kv: -kv[1])[:n_candidates]
        return {int(lab.split("_", 1)[1]) for lab, _ in items}
    cands = topc(sparse_scores) | (topc(dense_scores) if dense_scores else set())
    return sorted(cands)


def _ranked_labels(scores: dict[str, float], block_labels: list[int]) -> list[int]:
    """Rótulos do bloco ordenados pelo escore do recuperador (desc); ausentes ao fim."""
    sc = {int(l.split("_", 1)[1]): s for l, s in scores.items()}
    return sorted(block_labels, key=lambda c: -sc.get(c, float("-inf")))


@dataclass
class PackedBlocks:
    """Blocos empacotados block-diagonal (todos com TAMANHO UNIFORME) p/ 1 chamada."""
    n: int                                  # nº total de elementos (n_queries × block_size)
    block_size: int                         # M: tamanho uniforme de cada bloco (= L do método)
    label_lines: dict[int, list[int]]       # linhas-RK dos rótulos+fillers (cooc; compartilhadas)
    query_index: dict[str, int]             # qid → índice global de q
    local_map: dict[str, dict[int, int]]    # qid → {coluna_rótulo: índice global}
    index_label: dict[int, int]             # índice global → coluna do rótulo (só rótulos REAIS)
    query_block_labels: dict[str, list[int]]  # qid → rótulos do bloco
    query_all: dict[str, list[int]]         # qid → os M índices globais do bloco (p/ padding)


def _pad(line: list[int], pool: list[int], M: int) -> list[int]:
    """Completa `line` até comprimento M com elementos de `pool` ainda não usados
    (dentro do mesmo bloco → sem vazamento, sem duplicatas). Trunca se já passar."""
    if len(line) >= M:
        return line[:M]
    seen = set(line)
    out = list(line)
    for x in pool:
        if len(out) >= M:
            break
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out


def pack_blocks(qids: list[str], sparse: dict, dense: dict, Ytr: sp.csc_matrix,
                cfg: UdlfConfig) -> PackedBlocks:
    """Empacota os blocos de `qids` em faixas disjuntas, TODOS com o mesmo tamanho M
    (= 1 + maior nº de candidatos no lote). Blocos menores ganham índices de
    PREENCHIMENTO (fillers) — extras, dentro do bloco, NÃO mapeados a rótulo (somem no
    parsing) e fora do top-K real. Isso dá um arquivo RK de linhas uniformes (o binário
    exige) sem vazar entre blocos nem duplicar elementos numa linha."""
    per_q: list[tuple[str, list[int]]] = []
    for qid in qids:
        labels = block_candidates(sparse.get(qid, {}), dense.get(qid, {}), cfg.n_candidates)
        if labels:
            per_q.append((qid, labels))
    if not per_q:
        return PackedBlocks(0, 0, {}, {}, {}, {}, {}, {})
    M = max(1 + len(labels) for _, labels in per_q)      # tamanho uniforme do bloco

    g = 0
    label_lines: dict[int, list[int]] = {}
    query_index: dict[str, int] = {}
    local_map: dict[str, dict[int, int]] = {}
    index_label: dict[int, int] = {}
    query_block_labels: dict[str, list[int]] = {}
    query_all: dict[str, list[int]] = {}
    for qid, labels in per_q:
        gq = g
        m = len(labels)
        local = {c: gq + 1 + i for i, c in enumerate(labels)}
        fillers = list(range(gq + 1 + m, gq + M))        # M-1-m índices de preenchimento
        block_all = [gq] + [local[c] for c in labels] + fillers
        query_index[qid] = gq
        query_block_labels[qid] = labels
        local_map[qid] = local
        query_all[qid] = block_all
        for c, gi in local.items():
            index_label[gi] = c
        nbrs = block_label_neighbors(labels, Ytr)
        for c in labels:                                 # rótulo: vizinhos cooc + q + pad
            base = [local[c]] + [local[n] for n in nbrs[c]] + [gq]
            label_lines[local[c]] = _pad(base, block_all, M)
        for fi in fillers:                               # filler: self + resto (dropado depois)
            label_lines[fi] = _pad([fi], block_all, M)
        g = gq + M
    return PackedBlocks(n=g, block_size=M, label_lines=label_lines, query_index=query_index,
                        local_map=local_map, index_label=index_label,
                        query_block_labels=query_block_labels, query_all=query_all)


def assemble_lines(packed: PackedBlocks, scores_by_qid: dict) -> dict[int, list[int]]:
    """Linhas-RK completas = linhas (cooc) dos rótulos/fillers + a linha de cada q
    ranqueada pelos escores DESTE recuperador, preenchida ao tamanho M do bloco."""
    lines = dict(packed.label_lines)
    M = packed.block_size
    for qid, gq in packed.query_index.items():
        local = packed.local_map[qid]
        ranked = _ranked_labels(scores_by_qid.get(qid, {}), packed.query_block_labels[qid])
        base = [gq] + [local[c] for c in ranked]
        lines[gq] = _pad(base, packed.query_all[qid], M)
    return lines


def write_rk(packed: PackedBlocks, lines: dict[int, list[int]], path: str, topl: int) -> None:
    """Grava um arquivo RK/NUM: linha i = ranking (índices) do elemento i, top-L."""
    with open(path, "w", encoding="utf-8") as fh:
        for i in range(packed.n):
            row = lines.get(i, [i])
            if topl > 0:
                row = row[:topl]
            fh.write(" ".join(str(x) for x in row) + "\n")


def write_lists(packed: PackedBlocks, path: str) -> None:
    """lists file: 1 identificador por linha (índice → nome). NUM usa só a contagem."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(str(i) for i in range(packed.n)) + "\n")


def parse_output_to_runs(packed: PackedBlocks, out_rows: list[list[int]],
                         tag: str) -> dict[str, list[tuple[int, float]]]:
    """Da saída RK do UDLF, extrai a linha de cada q → [(coluna_rótulo, escore), ...]
    (escore sintético 1/(rank+1); só a ordem importa). Mantém só índices de rótulo."""
    runs: dict[str, list[tuple[int, float]]] = {}
    for qid, gq in packed.query_index.items():
        ranked = out_rows[gq] if gq < len(out_rows) else []
        items: list[tuple[int, float]] = []
        rank = 0
        for gi in ranked:
            col = packed.index_label.get(gi)
            if col is None:                      # pula q e quaisquer não-rótulos
                continue
            items.append((col, 1.0 / (rank + 1)))
            rank += 1
        runs[qid] = items
    return runs


# ─────────────────────────── execução (pyUDLF, import lazy) ─────────────────────────

def _run_udlf_batch(rk_paths: list[str], lists_path: str, n: int, block_size: int,
                    out_path: str, cfg: UdlfConfig) -> list[list[int]]:
    """Roda o pyUDLF (UDL se 1 arquivo, FUSION se >1) e devolve as linhas de saída
    (índices). Import lazy do pyUDLF — só aqui toca o binário.

    L = `block_size` (tamanho uniforme do bloco; o binário exige L ≤ N e linhas de
    comprimento L). K (kNN) é limitado a block_size-1."""
    from pyUDLF import run_calls as udlf
    from pyUDLF.utils import inputType

    m = UDLF_METHODS[cfg.method]
    L = cfg.label_topl if cfg.label_topl > 0 else block_size
    k_base = cfg.k_override or UDLF_DEFAULT_K[cfg.method]   # default do método, salvo override
    K = max(1, min(k_base, block_size - 1))                 # clamp só p/ bloco pequeno
    T = cfg.t_override or UDLF_DEFAULT_T[cfg.method]

    inp = inputType.InputType()
    if cfg.binary_path:
        inp.set_binary_path(cfg.binary_path)
    inp.set_method_name(m)
    inp.set_dataset_size(n)
    inp.set_lists_file(lists_path)
    inp.set_param("INPUT_FILE_FORMAT", "RK")
    inp.set_param("INPUT_RK_FORMAT", "NUM")
    inp.set_param("OUTPUT_FILE", "TRUE")
    inp.set_param("OUTPUT_FILE_FORMAT", "RK")
    inp.set_param("OUTPUT_RK_FORMAT", "NUM")
    inp.set_param("OUTPUT_FILE_PATH", out_path)
    inp.set_param("OUTPUT_LOG_FILE_PATH", out_path + "_log.txt")
    inp.set_param("EFFECTIVENESS_EVAL", "FALSE")
    inp.set_param("EFFICIENCY_EVAL", "FALSE")
    inp.set_param(f"PARAM_{m}_L", L)
    inp.set_param(f"PARAM_{m}_K", K)
    inp.set_param(f"PARAM_{m}_T", T)          # CPRR/LHRR/RFE têm o parâmetro T (iterações)
    if len(rk_paths) == 1:
        inp.set_param("UDL_TASK", "UDL")
        inp.set_input_files(rk_paths[0])
    else:
        inp.set_param("UDL_TASK", "FUSION")
        inp.set_input_files(rk_paths)
    res = udlf.run(inp, get_output=True)
    if res is False:
        raise RuntimeError(f"pyUDLF falhou (método {m}, n={n}) — ver log do UDLF")
    # pyUDLF expõe a saída ranqueada em res.rk_path (ver docs/udlf-integration.md);
    # fallback p/ o OUTPUT_FILE_PATH que pedimos, com variações de extensão.
    real = getattr(res, "rk_path", None)
    if not (real and os.path.isfile(real)):
        cands = [out_path, out_path + ".txt", out_path + ".rk"]
        real = next((c for c in cands if os.path.isfile(c)), None)
    if real is None:
        here = os.listdir(os.path.dirname(out_path) or ".")
        raise FileNotFoundError(
            f"saída do UDLF não encontrada — res.rk_path={getattr(res, 'rk_path', None)}; dir: {here}")
    with open(real, encoding="utf-8") as fh:
        return [[int(x) for x in line.split()] for line in fh if line.strip()]


# ─────────────────────────── orquestração por fold/lote ─────────────────────────────

def _batches(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def run_fold(fold, sparse: dict, dense: dict, udl_run: dict, Ytr, cfg: UdlfConfig,
             workdir: str) -> dict[str, list[tuple[int, float]]]:
    """Processa um fold em lotes de `block_batch` queries; devolve {qid: [(col, score)]}."""
    if cfg.mode == "fusion":
        qids = sorted(set(sparse) & set(dense), key=int)
    else:
        qids = sorted(set(udl_run), key=int)
    # agrupa queries de tamanho de bloco parecido por lote → minimiza fillers (padding)
    qids.sort(key=lambda q: len(block_candidates(sparse.get(q, {}), dense.get(q, {}),
                                                 cfg.n_candidates)))
    runs: dict[str, list[tuple[int, float]]] = {}
    for b, batch in enumerate(_batches(qids, cfg.block_batch)):
        packed = pack_blocks(batch, sparse, dense, Ytr, cfg)
        if packed.n == 0:
            continue
        lists_path = os.path.join(workdir, f"lists_{fold.fold_id}_{b}.txt")
        out_path = os.path.join(workdir, f"out_{fold.fold_id}_{b}")  # SEM .txt: o UDLF acrescenta
        write_lists(packed, lists_path)
        if cfg.mode == "fusion":
            rk_s = os.path.join(workdir, f"rk_s_{fold.fold_id}_{b}.txt")
            rk_d = os.path.join(workdir, f"rk_d_{fold.fold_id}_{b}.txt")
            write_rk(packed, assemble_lines(packed, sparse), rk_s, cfg.label_topl)
            write_rk(packed, assemble_lines(packed, dense), rk_d, cfg.label_topl)
            rk_paths = [rk_s, rk_d]
        else:
            rk = os.path.join(workdir, f"rk_u_{fold.fold_id}_{b}.txt")
            write_rk(packed, assemble_lines(packed, udl_run), rk, cfg.label_topl)
            rk_paths = [rk]
        out_rows = _run_udlf_batch(rk_paths, lists_path, packed.n, packed.block_size,
                                   out_path, cfg)
        runs.update(parse_output_to_runs(packed, out_rows, cfg.tag))
    return runs


def run_cv(cfg: UdlfConfig | None = None) -> None:
    """Roda o método UDLF (CPRR/LHRR/RFE) em cada fold e grava um run TREC por fold,
    consumível por metrics.py/gridsearch.py — lado a lado com a grade ranx."""
    import tempfile

    cfg = cfg or UdlfConfig()
    if cfg.method not in UDLF_METHODS:
        raise ValueError(f"método {cfg.method!r} desconhecido; use {sorted(UDLF_METHODS)}.")
    pooled = load_pooled(cfg.raw_dir)
    folds = make_folds(len(pooled), k=cfg.n_folds, seed=cfg.seed)
    print(f"UDLF {UDLF_METHODS[cfg.method]} | modo {cfg.mode} | {len(pooled)} docs "
          f"| rótulos {pooled.n_labels} | folds {list(cfg.fold_ids())} "
          f"| candidatos/bloco {cfg.n_candidates} | lote {cfg.block_batch}")

    workdir = tempfile.mkdtemp(prefix="udlf_run_")
    for fold in folds:
        if fold.fold_id not in cfg.fold_ids():
            continue
        sp_path = os.path.join(cfg.runs_dir, cfg.sparse_template.format(fold=fold.fold_id))
        de_path = os.path.join(cfg.runs_dir, cfg.dense_template.format(fold=fold.fold_id))
        sparse = run_to_dict(load_run(sp_path)) if os.path.exists(sp_path) else {}
        dense = run_to_dict(load_run(de_path)) if os.path.exists(de_path) else {}
        udl_run = {}
        if cfg.mode == "udl":
            up = os.path.join(cfg.runs_dir, cfg.udl_run_template.format(fold=fold.fold_id))
            udl_run = run_to_dict(load_run(up))
        Ytr = build_train_label_matrix(fold.train_idx, pooled.label_cols, pooled.n_labels)
        runs = run_fold(fold, sparse, dense, udl_run, Ytr, cfg, workdir)
        out = cfg.out_path(fold.fold_id)
        write_trec(runs, out, cfg.tag)
        print(f"  fold {fold.fold_id}: {len(runs)} queries → {out}")


def main(cfg: UdlfConfig | None = None) -> None:
    import argparse

    from src.data import add_dataset_arg, add_folds_arg, apply_dataset, parse_folds

    parser = argparse.ArgumentParser(description="Fusão/re-ranking contextual via UDLF")
    add_dataset_arg(parser)
    add_folds_arg(parser)
    parser.add_argument("--method", type=str, default=None, choices=sorted(UDLF_METHODS),
                        help="método UDLF (cprr/lhrr/rfe)")
    parser.add_argument("--mode", type=str, default=None, choices=["fusion", "udl"],
                        help="fusion = esparso+denso; udl = re-rank de 1 run fundido")
    parser.add_argument("--candidates", type=int, default=None, help="top-C por recuperador")
    parser.add_argument("--block-batch", type=int, default=None, help="queries por chamada do UDLF")
    args, _ = parser.parse_known_args()

    cfg = cfg or UdlfConfig()
    apply_dataset(cfg, args.dataset)
    cfg.folds = parse_folds(args.folds)
    if args.method:
        cfg.method = args.method
    if args.mode:
        cfg.mode = args.mode
    if args.candidates:
        cfg.n_candidates = args.candidates
    if args.block_batch:
        cfg.block_batch = args.block_batch
    run_cv(cfg)


if __name__ == "__main__":
    main()
