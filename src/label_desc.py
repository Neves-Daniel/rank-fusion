"""Geração de descrições de rótulo (RAG-labels) via LLM — réplica do `label_desc`
do RAG-Fuse (celsofranssa/RAG-Fuse), adaptada ao 5-fold CV e ao XMTC multi-rótulo.

Por que esta etapa: o recuperador denso do projeto é um bi-encoder label-as-document
(doc↔rótulo). O lado-rótulo precisa de um TEXTO por rótulo; em vez do nome cru do
EuroVoc, usamos uma descrição gerada por LLM a partir de exemplos contrastivos
(positivos = docs que têm o rótulo; negativos = docs que não têm). É o insumo que
destrava o treino do bi-encoder, gerado UMA vez e cacheado ("gerar uma vez,
iterar offline").

Fidelidade ao RAG-Fuse (task `label_desc`):
  - Prompt VERBATIM em src/prompts/label_desc_prompt.txt (mudá-lo muda a ciência).
  - Exemplo formatado como bloco `    text: <128 tokens> / labels: <nomes>`.
  - Parâmetros de geração: temperature=0.6, top_p=0.9, max_tokens=256,
    num_samples=5 positivos, e o mesmo modelo (llama-3.1-8b-instruct).

Divergências conscientes (registradas por honestidade):
  - **Multi-rótulo:** o RAG-Fuse assume 1 rótulo/doc; o EUR-Lex tem ~5,3. A linha
    `labels:` de cada exemplo lista TODOS os rótulos do doc.
  - **Backend:** o RAG-Fuse chama AWS Bedrock; aqui servimos o MESMO peso
    localmente via vLLM na Brev (só o cliente muda).
  - **Por fold:** as descrições são geradas por fold do 5-fold CV usando SÓ o
    corpus de treino do fold — evita que o doc-query influencie a descrição do
    rótulo usado para recuperá-lo (vazamento). O esparso e os folds usam a mesma
    seed (src/splits.make_folds), então os folds batem.
  - **Seleção reprodutível por rótulo:** o RNG da escolha de exemplos é semeado por
    (seed, fold_id, label_col), não por uma sequência global. Assim um run retomado
    (resume) produz exatamente as mesmas descrições de um run do zero.

Saída: um JSONL por fold em data/<ds>/rag-labels/fold{f}/labels_descriptions.jsonl,
uma linha `{"label_col", "label_name", "description"}` por rótulo. Inspecionável,
append-only (habilita resume), sem ABI de pickle.

NB: a geração com temperature>0 é só APROXIMADAMENTE reprodutível entre versões de
vLLM/hardware mesmo com seed fixa (mesma ressalva dos folds em src/splits.py).

Uso:
    python -m src.label_desc            # 5 folds, Eurlex-4K, modelo via vLLM (Brev)
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from importlib import resources
from typing import Callable, Protocol

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import _read_lines
from src.splits import Fold, PooledData, load_pooled, make_folds


def _load_prompt_template() -> str:
    """Carrega o template do prompt do resource (robusto ao cwd)."""
    return (resources.files("src.prompts") / "label_desc_prompt.txt").read_text(
        encoding="utf-8"
    )


PROMPT_TEMPLATE = _load_prompt_template()


@dataclass
class LabelDescConfig:
    raw_dir: str = "data/eurlex4k/raw"
    out_dir: str = "data/eurlex4k/rag-labels"          # 1 subpasta por fold aqui
    out_template: str = "fold{fold}/labels_descriptions.jsonl"
    # seleção de exemplos (fiel ao label_desc do RAG-Fuse)
    num_samples: int = 5           # positivos por rótulo
    num_negatives: int = 5         # negativos por rótulo
    text_max_tokens: int = 128     # primeiros N tokens (whitespace) de cada exemplo
    # geração (fiel ao label_desc do RAG-Fuse)
    # default = repo oficial (gated, exige aceite de licença + HF_TOKEN). Para testar
    # sem aprovação, exporte ANTES de rodar a env var com o espelho não-gated (mesmos
    # pesos): LABEL_DESC_MODEL=NousResearch/Meta-Llama-3.1-8B-Instruct
    model: str = os.environ.get("LABEL_DESC_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
    temperature: float = 0.6
    top_p: float = 0.9
    max_tokens: int = 256
    batch_size: int = 32           # prompts por chamada do backend / por gravação
    tensor_parallel_size: int = 1  # knob do vLLM p/ a caixa da Brev
    dtype: str = "auto"
    # protocolo / reprodutibilidade
    n_folds: int = 5               # = SparseConfig.n_folds (folds batem com o esparso)
    seed: int = 42                 # MESMA seed dos splits
    backend: str = "vllm"          # "vllm" em produção; testes injetam FakeLLM
    label_subset: int | None = None  # smoke-test: só os primeiros N rótulos
    resume: bool = True            # pula rótulos já gravados (idempotente)

    def fold_out_path(self, fold_id: int) -> str:
        return os.path.join(self.out_dir, self.out_template.format(fold=fold_id))


# ─────────────────────────── funções puras (testáveis) ───────────────────────────

def build_relevance_index(train_label_cols: list[list[int]]) -> dict[int, list[int]]:
    """Inverte os rótulos do corpus do fold: coluna de rótulo → posições locais
    (no corpus do fold) dos docs que a contêm. É o relevance_map do fold."""
    index: dict[int, list[int]] = {}
    for pos, cols in enumerate(train_label_cols):
        for c in cols:
            index.setdefault(c, []).append(pos)
    return index


def select_examples(
    label_col: int,
    relevance_index: dict[int, list[int]],
    n_docs: int,
    cfg: LabelDescConfig,
    rng: np.random.RandomState,
) -> tuple[list[int], list[int]]:
    """Escolhe posições locais de exemplos positivos e negativos para um rótulo.

    Positivos: docs do corpus do fold que TÊM o rótulo (até num_samples).
    Negativos: docs que NÃO têm o rótulo (até num_negatives), por amostragem com
    rejeição (negativos são quase todo o corpus, então é barato). Pode devolver
    menos negativos que o pedido se houver poucos disponíveis.

    Positivos pode vir VAZIO: rótulo de cauda ausente do treino deste fold. Nesse
    caso a descrição é gerada só com negativos + o nome do rótulo (ver generate_fold).
    """
    positives = relevance_index.get(label_col, [])
    if len(positives) > cfg.num_samples:
        positives = rng.choice(positives, size=cfg.num_samples, replace=False).tolist()
    else:
        positives = list(positives)

    pos_set = set(relevance_index.get(label_col, []))
    negatives: list[int] = []
    if len(pos_set) < n_docs:
        seen: set[int] = set()
        max_attempts = cfg.num_negatives * 20 + 50
        attempts = 0
        while len(negatives) < cfg.num_negatives and attempts < max_attempts:
            cand = int(rng.randint(0, n_docs))
            attempts += 1
            if cand in pos_set or cand in seen:
                continue
            seen.add(cand)
            negatives.append(cand)
    return positives, negatives


def format_example(text: str, label_names: list[str], cfg: LabelDescConfig) -> str:
    """Formata um exemplo no bloco exato do RAG-Fuse: texto truncado a
    `text_max_tokens` tokens (whitespace) + a lista de rótulos do doc, encerrado
    por linha em branco. (Multi-rótulo: lista TODOS os rótulos do doc.)"""
    truncated = " ".join(text.split()[: cfg.text_max_tokens])
    labels = ", ".join(label_names)
    return f"    text: {truncated}\n    labels: {labels}\n\n"


def build_prompt(
    target_label: str,
    pos_examples: list[str],
    neg_examples: list[str],
    template: str = PROMPT_TEMPLATE,
) -> str:
    """Monta o prompt final substituindo os placeholders do template verbatim.
    `pos_examples`/`neg_examples` são blocos já formatados por format_example."""
    return template.format(
        target_label=target_label,
        positive_texts_label_pairs="".join(pos_examples),
        negative_texts_label_pairs="".join(neg_examples),
    )


def is_numeric_label(name: str) -> bool:
    """True para rótulos cujo nome é só um código numérico do EuroVoc (ex.: '2164',
    '3485.0'), sem nome em linguagem natural. Usado só para diagnóstico/contagem."""
    try:
        float(name)
        return True
    except ValueError:
        return False


# ─────────────────────────── backends de LLM (swappable) ──────────────────────────

class LLMBackend(Protocol):
    def generate(self, prompts: list[str]) -> list[str]: ...


class VLLMBackend:
    """Backend de produção: serve o modelo localmente via vLLM (GPU/Brev).

    O import de vllm é LAZY (só aqui), espelhando o import tardio do retriv em
    src/retrieve_sparse.build_index — assim importar este módulo e rodar a suíte
    mínima nunca toca a stack de GPU.

    Usa `LLM.chat` (não `LLM.generate`): o chat template é aplicado internamente
    pelo vLLM, emitindo UM único BOS. Pré-renderizar com
    `tokenizer.apply_chat_template(tokenize=False)` e passar a string para
    `LLM.generate` duplicaria o `<|begin_of_text|>` — o generate offline força
    `add_special_tokens` (bug vllm#9519), e BOS duplo degrada silenciosamente a
    geração de modelos Llama, contaminando o insumo do recuperador denso.
    """

    def __init__(self, cfg: LabelDescConfig) -> None:
        from vllm import LLM, SamplingParams

        self.cfg = cfg
        self.llm = LLM(
            model=cfg.model,
            tensor_parallel_size=cfg.tensor_parallel_size,
            dtype=cfg.dtype,
        )
        self.sampling = SamplingParams(
            temperature=cfg.temperature,
            top_p=cfg.top_p,
            max_tokens=cfg.max_tokens,
            seed=cfg.seed,
        )

    def generate(self, prompts: list[str]) -> list[str]:
        convs = [[{"role": "user", "content": p}] for p in prompts]
        outputs = self.llm.chat(convs, self.sampling)  # template interno (1 BOS); ordem preservada
        return [o.outputs[0].text.strip() for o in outputs]


class FakeLLM:
    """Dublê de LLM para testes: determinístico, sem GPU, registra os prompts
    recebidos (análogo ao FakeSR em tests/test_retrieve_sparse.py)."""

    def __init__(self, fn: Callable[[str], str] | None = None) -> None:
        self.fn = fn or (lambda p: f"desc::{len(p)}")
        self.seen_prompts: list[str] = []

    def generate(self, prompts: list[str]) -> list[str]:
        self.seen_prompts.extend(prompts)
        return [self.fn(p) for p in prompts]


def make_backend(cfg: LabelDescConfig) -> LLMBackend:
    """Fábrica do backend. Mantém o caminho de GPU fora do import dos testes
    (que constroem FakeLLM diretamente)."""
    if cfg.backend == "vllm":
        return VLLMBackend(cfg)
    raise ValueError(
        f"backend {cfg.backend!r} não suportado pela fábrica; "
        "para testes, construa FakeLLM diretamente."
    )


# ─────────────────────────── I/O JSONL (append-only) ──────────────────────────────

def load_descriptions(path: str) -> dict[int, str]:
    """Lê o JSONL de descrições já geradas: {label_col: description}. Vazio se
    o arquivo não existe (resume começa do zero).

    Tolera uma última linha parcial/corrompida (processo morto no meio de um
    append): a linha inválida é ignorada — não está em `done`, então o rótulo é
    regerado no próximo run. Sem isso, um crash no momento errado tornaria o fold
    impossível de retomar (quebraria a garantia de resume idempotente)."""
    out: dict[int, str] = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue   # linha parcial de um run interrompido; será regerada
            out[int(obj["label_col"])] = obj["description"]
    return out


def append_descriptions(path: str, records: list[dict]) -> None:
    """Acrescenta linhas JSON ao arquivo (cria o diretório-pai se preciso).
    Faz flush+fsync por lote: a escrita fica durável em disco, encolhendo a
    janela em que um crash deixaria uma linha parcial para o resume."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        for obj in records:
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


# ─────────────────────────── geração por fold + orquestração ──────────────────────

def generate_fold(
    fold: Fold,
    pooled: PooledData,
    label_vocab: list[str],
    backend: LLMBackend,
    cfg: LabelDescConfig,
    template: str = PROMPT_TEMPLATE,
) -> dict[int, str]:
    """Gera (ou completa) as descrições de todos os rótulos de UM fold e grava no
    JSONL incrementalmente (resume-safe). Usa SÓ o corpus de treino do fold."""
    out_path = cfg.fold_out_path(fold.fold_id)

    train_texts = [pooled.texts[i] for i in fold.train_idx]
    train_label_cols = [pooled.label_cols[i] for i in fold.train_idx]
    n_docs = len(train_texts)

    relevance_index = build_relevance_index(train_label_cols)

    n_labels = pooled.n_labels
    if cfg.label_subset is not None:
        n_labels = min(cfg.label_subset, n_labels)

    done = load_descriptions(out_path) if cfg.resume else {}
    todo = [c for c in range(n_labels) if c not in done]
    results: dict[int, str] = {c: done[c] for c in done if c < n_labels}

    n_zero_pos = 0
    n_numeric = 0
    for start in range(0, len(todo), cfg.batch_size):
        batch_cols = todo[start : start + cfg.batch_size]
        prompts: list[str] = []
        for col in batch_cols:
            # RNG por rótulo: seleção independente da ordem → resume = run do zero
            rng = np.random.RandomState([cfg.seed, fold.fold_id, col])
            pos_idx, neg_idx = select_examples(col, relevance_index, n_docs, cfg, rng)
            if not pos_idx:
                n_zero_pos += 1
            target = label_vocab[col]
            if is_numeric_label(target):
                n_numeric += 1
            pos_blocks = [
                format_example(train_texts[i], [label_vocab[c] for c in train_label_cols[i]], cfg)
                for i in pos_idx
            ]
            neg_blocks = [
                format_example(train_texts[i], [label_vocab[c] for c in train_label_cols[i]], cfg)
                for i in neg_idx
            ]
            prompts.append(build_prompt(target, pos_blocks, neg_blocks, template))

        descriptions = backend.generate(prompts)
        records = [
            {"label_col": col, "label_name": label_vocab[col], "description": desc}
            for col, desc in zip(batch_cols, descriptions)
        ]
        append_descriptions(out_path, records)
        for rec in records:
            results[rec["label_col"]] = rec["description"]

    print(
        f"fold {fold.fold_id}: {len(todo)} gerados (+{len(done)} em cache) "
        f"| {n_zero_pos} sem positivos | {n_numeric} numéricos → {out_path}"
    )
    return results


def run_cv(cfg: LabelDescConfig | None = None) -> None:
    """Protocolo oficial: gera descrições por fold do 5-fold CV sobre o dataset
    agrupado. O backend (modelo) é construído UMA vez e reusado nos folds."""
    cfg = cfg or LabelDescConfig()
    pooled = load_pooled(cfg.raw_dir)
    label_vocab = _read_lines(os.path.join(cfg.raw_dir, "Y.txt"))
    if len(label_vocab) != pooled.n_labels:
        raise ValueError(
            f"Y.txt tem {len(label_vocab)} rótulos mas Y tem {pooled.n_labels} colunas."
        )

    n_gen = pooled.n_labels if cfg.label_subset is None else min(cfg.label_subset, pooled.n_labels)
    print(
        f"agrupado: {len(pooled)} docs | rótulos: {pooled.n_labels} (gerando {n_gen}) "
        f"| {cfg.n_folds}-fold (seed={cfg.seed}) | modelo: {cfg.model} via {cfg.backend}"
    )

    backend = make_backend(cfg)
    folds = make_folds(len(pooled), k=cfg.n_folds, seed=cfg.seed)
    for fold in folds:
        generate_fold(fold, pooled, label_vocab, backend, cfg)


def main(cfg: LabelDescConfig | None = None) -> None:
    run_cv(cfg)


if __name__ == "__main__":
    main()
