"""Testes de src/udlf_fusion.py — funções PURAS (blocos, RK, co-ocorrência, parsing).

A execução do binário UDLF (pyUDLF) NÃO é exercitada aqui (é opt-in, validada pelo
scripts/smoke_udlf.py). Importar o módulo não pode puxar o pyUDLF (import lazy).
"""
import subprocess
import sys

import numpy as np

from src.udlf_fusion import (
    UDLF_METHODS,
    UdlfConfig,
    assemble_lines,
    block_candidates,
    block_label_neighbors,
    build_train_label_matrix,
    pack_blocks,
    parse_output_to_runs,
)


def test_import_nao_puxa_pyudlf():
    code = (
        "import sys, src.udlf_fusion; "
        "assert 'pyUDLF' not in sys.modules, 'pyUDLF importado no load do módulo'; "
        "print('ok')"
    )
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
    assert "ok" in res.stdout


def test_block_candidates_uniao_topc():
    sparse = {"label_1": 0.9, "label_2": 0.5, "label_3": 0.1}
    dense = {"label_2": 0.8, "label_4": 0.7}
    # top-2 esparso = {1,2}; top-2 denso = {2,4}; união = {1,2,4}
    assert block_candidates(sparse, dense, n_candidates=2) == [1, 2, 4]


def test_apply_dataset_nao_clobbera_metodo():
    from src.data import apply_dataset
    cfg = UdlfConfig()
    apply_dataset(cfg, "wiki10-31k")
    assert cfg.raw_dir == "data/wiki10-31k/raw"
    assert cfg.runs_dir == "data/wiki10-31k/runs"
    assert callable(cfg.out_path)               # continua método, não virou string
    assert cfg.method == "cprr" and cfg.mode == "fusion"


def test_build_train_label_matrix_coocorrencia():
    # 3 docs de treino, 4 rótulos. label_cols por doc GLOBAL.
    label_cols = [[0, 1], [1, 2], [0, 1], [3]]   # docs 0..3
    train_idx = np.array([0, 1, 2])              # treino = docs 0,1,2 (exclui o 3)
    Ytr = build_train_label_matrix(train_idx, label_cols, n_labels=4)
    assert Ytr.shape == (3, 4)
    # rótulo 1 aparece nos 3 docs de treino; rótulo 3 em nenhum (era do doc 3, fora)
    assert Ytr[:, 1].sum() == 3
    assert Ytr[:, 3].sum() == 0


def test_block_label_neighbors_ordena_por_cooc():
    # rótulos 0 e 1 co-ocorrem sempre; 2 isolado → vizinho mais próximo de 0 é 1
    label_cols = [[0, 1], [0, 1], [2]]
    Ytr = build_train_label_matrix(np.array([0, 1, 2]), label_cols, n_labels=3)
    nbrs = block_label_neighbors([0, 1, 2], Ytr)
    assert nbrs[0][0] == 1                       # 1 é o vizinho top de 0
    assert nbrs[1][0] == 0


def test_pack_blocks_faixas_disjuntas():
    sparse = {"5": {"label_0": 0.9, "label_1": 0.5}, "7": {"label_0": 0.8, "label_2": 0.4}}
    dense = {"5": {"label_1": 0.7}, "7": {"label_2": 0.6}}
    label_cols = [[0, 1]] * 10
    Ytr = build_train_label_matrix(np.arange(10), label_cols, n_labels=3)
    cfg = UdlfConfig(n_candidates=5)
    packed = pack_blocks(["5", "7"], sparse, dense, Ytr, cfg)
    # q5: índice 0, rótulos {0,1} em 1,2 → bloco [0..2]; q7: índice 3, rótulos {0,2} em 4,5
    assert packed.query_index == {"5": 0, "7": 3}
    assert packed.n == 6
    # índices de rótulo mapeiam de volta à coluna certa, sem colisão entre blocos
    assert packed.index_label[1] in (0, 1) and packed.index_label[4] in (0, 2)
    assert set(packed.index_label) == {1, 2, 4, 5}   # só índices de rótulo (não os q: 0,3)


def test_assemble_lines_q_por_escore():
    sparse = {"5": {"label_0": 0.1, "label_1": 0.9}}   # rótulo 1 ranqueado acima do 0
    dense = {"5": {"label_0": 0.5, "label_1": 0.5}}
    Ytr = build_train_label_matrix(np.arange(4), [[0, 1]] * 4, n_labels=2)
    packed = pack_blocks(["5"], sparse, dense, Ytr, UdlfConfig(n_candidates=5))
    lines = assemble_lines(packed, sparse)
    gq = packed.query_index["5"]
    local = packed.local_map["5"]
    # linha de q começa com q e depois os rótulos por escore desc (label_1 antes de label_0)
    assert lines[gq][0] == gq
    assert lines[gq][1] == local[1] and lines[gq][2] == local[0]


def test_parse_output_dropa_q_e_gera_score_sintetico():
    sparse = {"5": {"label_0": 0.9, "label_1": 0.5}}
    dense = {"5": {"label_1": 0.7}}
    Ytr = build_train_label_matrix(np.arange(4), [[0, 1]] * 4, n_labels=2)
    packed = pack_blocks(["5"], sparse, dense, Ytr, UdlfConfig(n_candidates=5))
    gq = packed.query_index["5"]
    local = packed.local_map["5"]
    # saída simulada do UDLF p/ a linha de q: [q, rótulo_col1, rótulo_col0]
    out_rows = [[] for _ in range(packed.n)]
    out_rows[gq] = [gq, local[1], local[0]]
    runs = parse_output_to_runs(packed, out_rows, "udlf")
    cols = [c for c, _ in runs["5"]]
    assert cols == [1, 0]                         # q removido; ordem preservada
    assert runs["5"][0][1] > runs["5"][1][1]      # score sintético 1/(rank+1) decrescente


def test_udlf_methods_mapeia_os_tres():
    assert set(UDLF_METHODS) == {"cprr", "lhrr", "rfe"}


def test_pack_blocks_uniforme_com_fillers():
    # q1 tem 3 candidatos, q2 tem 1 → blocos devem ficar do MESMO tamanho (padding)
    sparse = {"1": {"label_0": .9, "label_1": .8, "label_2": .7}, "2": {"label_0": .9}}
    dense = {"1": {}, "2": {}}
    Ytr = build_train_label_matrix(np.arange(6), [[0, 1, 2]] * 6, n_labels=3)
    packed = pack_blocks(["1", "2"], sparse, dense, Ytr, UdlfConfig(n_candidates=5))
    M = packed.block_size
    assert M == 4                                  # 1 (q) + 3 candidatos (o maior bloco)
    assert packed.n == 2 * M                       # ambos os blocos padded a M
    # todas as linhas (de TODOS os índices) têm exatamente M elementos → arquivo uniforme
    lines = assemble_lines(packed, sparse)
    assert all(len(lines[i]) == M for i in range(packed.n))
    # nenhuma linha vaza pro outro bloco (índices < M no bloco 0, ≥ M no bloco 1)
    for i in range(packed.n):
        blk = set(range(0, M)) if i < M else set(range(M, 2 * M))
        assert set(lines[i]) <= blk
    # q2 só tem 1 rótulo real → os demais índices do bloco são fillers (não viram saída)
    assert sum(1 for gi in range(M, 2 * M) if gi in packed.index_label) == 1
