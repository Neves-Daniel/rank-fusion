"""Testes do gerador de RAG-labels (src/label_desc.py).

Tudo aqui roda na CPU, em segundos: as funções de seleção/formatação/prompt são
puras e a geração usa o dublê FakeLLM (sem GPU, sem vllm). O smoke test do backend
vLLM real fica em tests/test_label_desc_real_vllm.py (opt-in: -m vllm).
"""
import json
import pathlib
import subprocess
import sys

import numpy as np

from src.label_desc import (
    FakeLLM,
    LabelDescConfig,
    append_descriptions,
    build_prompt,
    build_relevance_index,
    format_example,
    generate_fold,
    is_numeric_label,
    load_descriptions,
    select_examples,
)
from src.splits import Fold, PooledData


def _toy_pooled() -> tuple[PooledData, list[str]]:
    """5 docs, 4 rótulos. col0 é numérico (de propósito); col3 só aparece no doc de
    teste (doc4), logo tem ZERO positivos no treino [0..3]."""
    texts = [
        "fish ocean fishery boats nets",     # doc0 -> cols 0,1
        "wheat farm crops harvest soil",      # doc1 -> cols 1,2
        "bank euro currency monetary",        # doc2 -> cols 0,2
        "court law justice legal rights",     # doc3 -> col 2
        "space rocket satellite orbit",       # doc4 -> col 3 (só teste)
    ]
    label_cols = [[0, 1], [1, 2], [0, 2], [2], [3]]
    vocab = ["2164", "fisheries", "agriculture", "space policy"]
    return PooledData(texts=texts, label_cols=label_cols, n_labels=4), vocab


# ───────────────────────────── funções puras ─────────────────────────────

def test_build_relevance_index_inverte_rotulos():
    rel = build_relevance_index([[0, 1], [1, 2], [0, 2], [2]])
    assert rel[0] == [0, 2]
    assert rel[1] == [0, 1]
    assert rel[2] == [1, 2, 3]
    assert 3 not in rel


def test_format_example_trunca_e_lista_todos_os_rotulos():
    cfg = LabelDescConfig(text_max_tokens=5)
    block = format_example("a b c d e f g h", ["lab x", "lab y"], cfg)
    assert "text: a b c d e" in block
    assert "f g h" not in block                  # truncado em 5 tokens
    assert "labels: lab x, lab y" in block        # multi-rótulo: lista todos
    assert block.endswith("\n\n")


def test_build_prompt_substitui_placeholders_e_target_numerico():
    pos = ["    text: foo\n    labels: A\n\n"]
    neg = ["    text: bar\n    labels: B\n\n"]
    prompt = build_prompt("2164", pos, neg)
    assert '"2164"' in prompt                      # rótulo numérico passa verbatim
    assert prompt.rstrip().endswith("2164:")       # cue final de geração
    assert "    text: foo\n    labels: A" in prompt
    assert "    text: bar\n    labels: B" in prompt
    assert "Positive examples" in prompt and "Negative examples" in prompt
    # blocos caem na seção CORRETA (não basta estarem presentes): pos antes de neg.
    # Pega uma eventual troca dos placeholders {positive_...} <-> {negative_...}.
    assert (prompt.index("Positive examples") < prompt.index("foo")
            < prompt.index("Negative examples") < prompt.index("bar"))
    assert "{target_label}" not in prompt          # placeholders resolvidos
    assert "{positive_texts_label_pairs}" not in prompt


def test_is_numeric_label():
    assert is_numeric_label("2164")
    assert is_numeric_label("3485.0")
    assert not is_numeric_label("fisheries")
    assert not is_numeric_label("abolition of customs duties")


def test_select_examples_positivos_tem_o_rotulo_negativos_nao():
    _, _vocab = _toy_pooled()
    tlc = [[0, 1], [1, 2], [0, 2], [2]]
    rel = build_relevance_index(tlc)
    cfg = LabelDescConfig(num_samples=2, num_negatives=2)
    pos, neg = select_examples(2, rel, len(tlc), cfg, np.random.RandomState([42, 0, 2]))
    assert pos and all(2 in tlc[i] for i in pos)
    assert all(2 not in tlc[i] for i in neg)


def test_select_examples_seeded_deterministico_e_sensivel_a_seed():
    # corpus com MAIS positivos que num_samples e mais não-positivos que num_negatives
    # → rng.choice de fato roda e a seed dirige a seleção (mesma seed = igual;
    # seed diferente = diferente). Um RNG ignorado passaria só no 1º assert.
    tlc = [[0]] * 8 + [[1]] * 8
    rel = build_relevance_index(tlc)
    cfg = LabelDescConfig(num_samples=3, num_negatives=3)
    a = select_examples(0, rel, len(tlc), cfg, np.random.RandomState([42, 0, 0]))
    b = select_examples(0, rel, len(tlc), cfg, np.random.RandomState([42, 0, 0]))
    c = select_examples(0, rel, len(tlc), cfg, np.random.RandomState([99, 0, 0]))
    assert a == b                      # mesma seed → mesmo resultado
    assert a != c                      # seed diferente → seleção diferente (RNG dirige)
    assert len(a[0]) == 3              # amostrou 3 positivos (rng.choice rodou)
    assert all(0 in tlc[i] for i in a[0]) and all(0 not in tlc[i] for i in a[1])


def test_select_examples_rotulo_de_cauda_sem_positivos():
    tlc = [[0, 1], [1, 2], [0, 2], [2]]          # col 3 não aparece
    rel = build_relevance_index(tlc)
    cfg = LabelDescConfig(num_samples=2, num_negatives=2)
    pos, neg = select_examples(3, rel, len(tlc), cfg, np.random.RandomState([42, 0, 3]))
    assert pos == []                              # sem positivos
    assert len(neg) == 2 and all(3 not in tlc[i] for i in neg)


def test_select_examples_corpus_todo_positivo_nao_trava():
    tlc = [[0], [0], [0]]                          # todos os docs têm o rótulo 0
    rel = build_relevance_index(tlc)
    cfg = LabelDescConfig(num_samples=2, num_negatives=5)
    pos, neg = select_examples(0, rel, len(tlc), cfg, np.random.RandomState([42, 0, 0]))
    assert neg == []                              # sem negativos disponíveis (sem loop infinito)
    assert set(pos) <= {0, 1, 2}


def test_select_examples_poucos_negativos_devolve_menos():
    tlc = [[0], [0], [0], [1]]                     # só o doc 3 é negativo p/ o rótulo 0
    rel = build_relevance_index(tlc)
    cfg = LabelDescConfig(num_samples=2, num_negatives=5)
    pos, neg = select_examples(0, rel, len(tlc), cfg, np.random.RandomState([42, 0, 0]))
    assert neg == [3]                             # devolve menos que num_negatives, sem travar
    assert 0 not in tlc[neg[0]]


# ───────────────────────────── I/O JSONL ─────────────────────────────

def test_append_load_roundtrip_cria_diretorio_pai(tmp_path):
    path = str(tmp_path / "sub" / "desc.jsonl")    # pai ainda não existe
    append_descriptions(path, [{"label_col": 5, "label_name": "x", "description": "d5"}])
    append_descriptions(path, [{"label_col": 7, "label_name": "y", "description": "d7"}])
    assert load_descriptions(path) == {5: "d5", 7: "d7"}


def test_load_descriptions_arquivo_inexistente(tmp_path):
    assert load_descriptions(str(tmp_path / "nope.jsonl")) == {}


def test_load_descriptions_ignora_linha_truncada(tmp_path):
    # resume após crash: última linha parcial (sem newline) NÃO deve quebrar a leitura
    path = str(tmp_path / "d.jsonl")
    append_descriptions(path, [{"label_col": 0, "label_name": "a", "description": "d0"}])
    with open(path, "a", encoding="utf-8") as fh:
        fh.write('{"label_col": 1, "label_name": "b", "desc')   # linha truncada
    assert load_descriptions(path) == {0: "d0"}   # linha boa lida; truncada ignorada (será regerada)


# ───────────────────────── geração ponta-a-ponta (FakeLLM) ─────────────────────────

def test_generate_fold_grava_todos_os_rotulos(tmp_path):
    pooled, vocab = _toy_pooled()
    fold = Fold(fold_id=0, train_idx=np.array([0, 1, 2, 3]), test_idx=np.array([4]))
    cfg = LabelDescConfig(out_dir=str(tmp_path), batch_size=2, num_samples=2, num_negatives=2)
    fake = FakeLLM(lambda p: "uma descrição")
    res = generate_fold(fold, pooled, vocab, fake, cfg)

    assert set(res.keys()) == {0, 1, 2, 3}
    assert len(fake.seen_prompts) == 4             # 1 prompt por rótulo
    desc = load_descriptions(cfg.fold_out_path(0))
    assert set(desc.keys()) == {0, 1, 2, 3}
    assert all(isinstance(k, int) for k in desc)
    # col3 (cauda sem positivos no treino) também recebeu descrição
    assert desc[3] == "uma descrição"
    # cada prompt termina com o cue do rótulo-alvo correto (montagem certa por coluna)
    cues = {p.rstrip().splitlines()[-1] for p in fake.seen_prompts}
    assert cues == {f"{vocab[c]}:" for c in range(4)}


def test_generate_fold_resume_pula_cacheados(tmp_path):
    pooled, vocab = _toy_pooled()
    fold = Fold(fold_id=0, train_idx=np.array([0, 1, 2, 3]), test_idx=np.array([4]))
    cfg = LabelDescConfig(out_dir=str(tmp_path), batch_size=2, num_samples=2, num_negatives=2, resume=True)
    out = cfg.fold_out_path(0)
    # pré-grava 2 rótulos (run anterior interrompido)
    append_descriptions(out, [
        {"label_col": 0, "label_name": vocab[0], "description": "cacheado-0"},
        {"label_col": 1, "label_name": vocab[1], "description": "cacheado-1"},
    ])
    fake = FakeLLM()
    generate_fold(fold, pooled, vocab, fake, cfg)

    assert len(fake.seen_prompts) == pooled.n_labels - 2   # só os restantes
    # nenhuma duplicata: 1 linha por rótulo
    lines = [l for l in open(out, encoding="utf-8") if l.strip()]
    assert len(lines) == pooled.n_labels
    desc = load_descriptions(out)
    assert len(desc) == pooled.n_labels
    assert desc[0] == "cacheado-0" and desc[1] == "cacheado-1"   # cache preservado


def test_resume_produz_descricoes_identicas_a_run_do_zero(tmp_path):
    # Garantia central do módulo: como o RNG é semeado por (seed, fold_id, col), a
    # seleção independe da ordem/batch e do resume → um run retomado gera EXATAMENTE
    # o mesmo de um run do zero. FakeLLM(fn=p->p) torna a descrição == o próprio
    # prompt, então comparar os dicts compara a montagem exata. Uma regressão p/ um
    # RNG global/sequencial quebraria isto (a ordem de consumo do RNG mudaria).
    pooled, vocab = _toy_pooled()
    fold = Fold(fold_id=0, train_idx=np.array([0, 1, 2, 3]), test_idx=np.array([4]))
    echo = lambda p: p

    cfg_scratch = LabelDescConfig(out_dir=str(tmp_path / "scratch"), batch_size=8)
    scratch = generate_fold(fold, pooled, vocab, FakeLLM(echo), cfg_scratch)

    cfg_resume = LabelDescConfig(out_dir=str(tmp_path / "resume"), batch_size=1, resume=True)
    out = cfg_resume.fold_out_path(0)
    append_descriptions(out, [   # pré-cacheia 2 cols (run anterior interrompido)
        {"label_col": 0, "label_name": vocab[0], "description": scratch[0]},
        {"label_col": 2, "label_name": vocab[2], "description": scratch[2]},
    ])
    resumed = generate_fold(fold, pooled, vocab, FakeLLM(echo), cfg_resume)

    assert resumed == scratch    # cols cacheados + regerados == run do zero (batch diferente)


def test_sem_vazamento_doc_de_teste_nao_entra_nos_prompts(tmp_path):
    # Decisão travada: a descrição de um fold usa SÓ o corpus de treino do fold.
    # train_idx é NÃO-identidade (global != local) p/ pegar uma regressão que
    # indexasse pooled.texts por posição local como se fosse global.
    texts = [
        "TESTDOC unique LEAKMARKER fish ocean",   # doc0 = QUERY/teste (test_idx)
        "wheat farm crops harvest",                # doc1 treino (col 1)
        "bank euro currency monetary",             # doc2 treino (col 1, compartilha rótulo c/ doc0)
        "court law justice legal",                 # doc3 treino (col 2)
        "space rocket satellite orbit",            # doc4 treino (col 0)
    ]
    label_cols = [[1], [1], [1], [2], [0]]         # doc0 (teste) compartilha col1 com docs 1,2
    vocab = ["c0", "c1", "c2"]
    pooled = PooledData(texts=texts, label_cols=label_cols, n_labels=3)
    fold = Fold(fold_id=0, train_idx=np.array([1, 2, 3, 4]), test_idx=np.array([0]))
    cfg = LabelDescConfig(out_dir=str(tmp_path), batch_size=8)
    fake = FakeLLM()
    generate_fold(fold, pooled, vocab, fake, cfg)

    blob = "\n".join(fake.seen_prompts)
    assert "LEAKMARKER" not in blob    # texto do doc de teste NUNCA entra num prompt


def test_label_subset_limita_geracao(tmp_path):
    pooled, vocab = _toy_pooled()
    fold = Fold(fold_id=0, train_idx=np.array([0, 1, 2, 3]), test_idx=np.array([4]))
    cfg = LabelDescConfig(out_dir=str(tmp_path), batch_size=8, label_subset=2)
    fake = FakeLLM()
    res = generate_fold(fold, pooled, vocab, fake, cfg)
    assert set(res.keys()) == {0, 1}               # só os 2 primeiros rótulos


# ───────────────────────── import é CPU-safe (vllm lazy) ─────────────────────────

def test_import_do_modulo_nao_carrega_vllm():
    """Importar src.label_desc não deve puxar vllm (import só no VLLMBackend).
    Roda em subprocesso p/ não depender da ordem de coleta do pytest."""
    root = pathlib.Path(__file__).resolve().parents[1]
    code = "import sys, src.label_desc; assert 'vllm' not in sys.modules"
    res = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True)
    assert res.returncode == 0, res.stderr
