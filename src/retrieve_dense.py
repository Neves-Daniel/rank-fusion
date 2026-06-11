"""Recuperador denso (bi-encoder BERT fine-tuned) para XMTC — porte fiel do
`DenseRetriever` do RAG-Fuse (celsofranssa/RAG-Fuse), adaptado ao 5-fold CV.

Abordagem (label-as-document): um BERT fine-tunado mapeia DOCUMENTO e RÓTULO num
mesmo espaço vetorial; cada rótulo é tratado como um pequeno "documento" cujo texto
é `f"{nome_do_rótulo} {descrição}"`. Recupera-se rótulos por similaridade direta
doc×rótulo.
  1. Resolve o texto de cada rótulo (ver `label_enhancement`).
  2. Treina o encoder com perda contrastiva NT-Xent (temp 0.07) + um miner que usa
     o relevance-map (quais rótulos são gold de cada doc) para marcar pares
     positivos/negativos DENTRO do batch. Só o corpus de treino do fold é usado.
  3. Inferência: embeda as queries (docs de teste do fold) e TODOS os rótulos,
     calcula similaridade cosine exata e mantém os top-`num_labels` rótulos de
     CABEÇA + top-`num_labels` de CAUDA (64+64=128) por query.
  4. Exporta um run TREC por fold (`qid Q0 label_id rank score tag`), consumido
     depois por fusion.py / metrics.py — mesmo formato do esparso.

Fidelidade ao RAG-Fuse:
  - Encoder = `bert-base-uncased` (output_hidden_states) + ConcatenatePooling:
    concatena as 4 últimas camadas no token [CLS] → 3072-d, L2-normalizado.
  - Loss = `pytorch_metric_learning.losses.NTXentLoss(temperature=0.07,
    distance=DotProductDistance(escala 20))` + `RelevanceMiner` (porte do BaseMiner).
  - Otimização = AdamW(lr=5e-5, wd=1e-2, amsgrad) + warmup linear (config ATIVA do
    RAG-Fuse; o CyclicLR de lá está comentado), 5 épocas, fp16.

Divergências conscientes (registradas por honestidade):
  - **Sem PyTorch-Lightning/Hydra/nmslib:** torch+transformers puros, no estilo do
    porte do esparso (retriv) em src/retrieve_sparse.py. A busca ANN por HNSW é
    trocada por similaridade EXATA (matmul cosine): com ~4k rótulos é barato, exato
    e mais reprodutível.
  - **Por fold:** treino e descrições de rótulo (RAG-labels) usam SÓ o corpus de
    treino do fold (mesma seed dos splits) — sem vazamento test→modelo/descrição.
  - **RAG-labels OPCIONAL** via `label_enhancement` (knob nativo do RAG-Fuse):
    "LLM" usa as descrições geradas em src/label_desc.py; "NONE" cai no nome cru do
    EuroVoc (fallback automático se o JSONL do fold não existir).

NB: imports pesados (torch/transformers/pytorch-metric-learning) são LAZY — importar
este módulo e rodar a suíte mínima de testes (FakeEncoder) NÃO toca a stack de GPU,
espelhando o import tardio do retriv/vllm nos outros recuperadores.

Uso (na Brev/GPU):
    python -m src.retrieve_dense                 # 5 folds; pula os que já têm .trec (resume)
    python -m src.retrieve_dense --fold 3        # só o fold 3 (resume direcionado)
    python -m src.retrieve_dense --fold 4 --device cuda:1   # paralelizar folds em GPUs
    python -m src.retrieve_dense --no-resume     # refaz tudo, ignorando .trec existentes
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np

if __package__ in {None, ""}:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import _read_lines
from src.label_desc import load_descriptions
from src.retrieve_sparse import head_tail_split, write_trec
from src.splits import Fold, PooledData, load_pooled, make_folds


@dataclass
class DenseConfig:
    # caminhos
    raw_dir: str = "data/eurlex4k/raw"
    runs_dir: str = "data/eurlex4k/runs"               # 1 run por fold aqui
    out_template: str = "dense.fold{fold}.trec"        # nome do run de cada fold
    rag_labels_dir: str = "data/eurlex4k/rag-labels"   # saída do src/label_desc.py
    rag_labels_template: str = "fold{fold}/labels_descriptions.jsonl"
    # encoder / tokenização (fiel ao RAG-Fuse)
    architecture: str = "bert-base-uncased"
    text_max_length: int = 512        # docs longos da EUR-Lex truncados (ameaça à validade)
    label_max_length: int = 256       # cabe a descrição RAG-labels (gen max_tokens=256)
    # treino contrastivo (config ATIVA do RAG-Fuse)
    epochs: int = 5
    # batch_size = 32 (fiel ao RAG-Fuse). Mantido para os 5 folds do CV serem
    # treinados IDENTICAMENTE — mudar o batch muda o nº de negativos in-batch e de
    # passos/época (confunde efeito-de-fold com efeito-de-hiperparâmetro). Para
    # acelerar sem mexer na fidelidade, paralelize folds em GPUs (--fold/--device),
    # não aumente o batch.
    batch_size: int = 32
    lr: float = 5e-5
    weight_decay: float = 1e-2
    warmup_ratio: float = 0.0         # RAG-Fuse: num_warmup_steps=0
    temperature: float = 0.07         # NT-Xent
    precision: str = "fp16"           # "fp16" (autocast+GradScaler) em GPU; ignorado em CPU
    num_workers: int = 8              # caixa do lab tem 16 CPUs; evita a GPU passar fome (fidelidade-neutro)
    # inferência / split cabeça-cauda
    num_labels: int = 64              # rótulos mantidos POR classe (64+64=128, como o artigo)
    head_frac: float = 0.20           # Pareto: 20% mais frequentes = cabeça
    encode_batch_size: int = 256      # lote de embedding na inferência
    # inferência chunkada (vocabulários enormes tipo Amazon-670K): a matriz [Nq, Nl]
    # do cosine não materializa (670K rótulos × ~128K queries seria >300 GB). Acima de
    # infer_chunk_threshold rótulos, ranqueia em blocos de query (rótulos no device uma
    # vez) — resultado IDÊNTICO ao rank_per_class. Abaixo, segue o numpy exato (intacto).
    infer_chunk_threshold: int = 100_000
    infer_chunk_size: int = 1024      # queries por bloco no scorer chunkado
    # opcionalidade das RAG-labels (knob nativo do RAG-Fuse)
    label_enhancement: str = "LLM"    # "LLM" = descrições; "NONE" = só o nome cru
    # protocolo / reprodutibilidade
    n_folds: int = 5                  # = SparseConfig/LabelDescConfig (folds batem)
    seed: int = 42                    # MESMA seed dos splits/label_desc
    device: str = "cuda"
    tag: str = "dense"
    resume: bool = True               # pula fold cujo run .trec já existe (idempotente)

    def fold_out_path(self, fold_id: int) -> str:
        return os.path.join(self.runs_dir, self.out_template.format(fold=fold_id))

    def fold_rag_path(self, fold_id: int) -> str:
        return os.path.join(self.rag_labels_dir, self.rag_labels_template.format(fold=fold_id))


# ─────────────────────────── funções puras (testáveis na CPU) ──────────────────────

def resolve_label_texts(
    label_vocab: list[str],
    descriptions: dict[int, str],
    label_enhancement: str,
) -> list[str]:
    """Texto de cada rótulo (lado-rótulo do bi-encoder), fiel ao RAG-Fuse: sempre
    `f"{nome} {features}"`, com `features` = descrição RAG-labels ("LLM") ou vazio
    ("NONE"). É o SEAM que torna as RAG-labels opcionais: `descriptions` vazio
    (arquivo do fold ausente, ou label_enhancement="NONE") → fallback ao nome cru.

    Rótulo sem descrição (cauda ausente do treino do fold, ou só código numérico)
    cai no nome cru — todo `col` sempre tem um texto, então o encoder nunca acha
    chave faltando.
    """
    if label_enhancement == "LLM":
        return [f"{name} {descriptions.get(col, '')}".strip() for col, name in enumerate(label_vocab)]
    if label_enhancement == "NONE":
        return [name.strip() for name in label_vocab]
    raise ValueError(f"label_enhancement deve ser 'LLM' ou 'NONE', recebido {label_enhancement!r}.")


def build_relevance_map(label_cols: list[list[int]]) -> dict[int, set[int]]:
    """{índice global do doc → conjunto de colunas-gold}. Usado pelo miner (marca
    pares positivos no batch) e como base do split cabeça/cauda."""
    return {i: set(cols) for i, cols in enumerate(label_cols)}


def build_train_pairs(fold: Fold, pooled: PooledData) -> list[tuple[int, int]]:
    """Explode pares positivos `(idx_global_do_doc, coluna_do_rótulo)` do corpus de
    treino do fold — um par por rótulo-gold de cada doc (porte do RetrieverFitDataset).
    O `idx_global` é a chave do relevance_map; o texto é `pooled.texts[idx_global]`."""
    pairs: list[tuple[int, int]] = []
    for idx in fold.train_idx.tolist():
        for col in pooled.label_cols[idx]:
            pairs.append((idx, col))
    return pairs


def _topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    """Índices dos `k` maiores valores de `scores` (1-D), em ordem decrescente.
    argpartition (O(n)) + ordena só os k selecionados."""
    k = min(k, scores.shape[0])
    if k <= 0:
        return np.empty(0, dtype=np.intp)
    part = np.argpartition(-scores, k - 1)[:k]
    return part[np.argsort(-scores[part])]


def rank_per_class(
    text_rpr: np.ndarray,
    query_ids: list[str],
    label_rpr: np.ndarray,
    label_ids: list[int],
    head: set[int],
    tail: set[int],
    num_labels: int,
) -> dict[str, list[tuple[int, float]]]:
    """Ranking por similaridade cosine exata, mantendo top-`num_labels` rótulos de
    CABEÇA + top-`num_labels` de CAUDA por query (64+64). Vetores já vêm
    L2-normalizados (ConcatenatePooling), então o produto interno É o cosseno.

    Retorna {qid: [(coluna_rótulo, score), ...]} com até num_labels*2 candidatos —
    mesmo formato consumido por write_trec.
    """
    label_ids_arr = np.asarray(label_ids)
    head_pos = np.array([i for i, c in enumerate(label_ids) if c in head], dtype=np.intp)
    tail_pos = np.array([i for i, c in enumerate(label_ids) if c in tail], dtype=np.intp)

    sims = text_rpr @ label_rpr.T   # [Nq, Nl] cosine
    runs: dict[str, list[tuple[int, float]]] = {}
    for qi, qid in enumerate(query_ids):
        row = sims[qi]
        items: list[tuple[int, float]] = []
        for pos in (head_pos, tail_pos):
            if pos.shape[0] == 0:
                continue
            sub = row[pos]
            top = _topk_indices(sub, num_labels)
            cols = label_ids_arr[pos[top]]
            items.extend((int(c), float(s)) for c, s in zip(cols, sub[top]))
        runs[qid] = items
    return runs


def rank_per_class_chunked(
    text_rpr: np.ndarray,
    query_ids: list[str],
    label_rpr: np.ndarray,
    label_ids: list[int],
    head: set[int],
    tail: set[int],
    num_labels: int,
    chunk_size: int = 1024,
    device: str = "cuda",
) -> dict[str, list[tuple[int, float]]]:
    """Versão chunkada de `rank_per_class` para vocabulários enormes (ex.: Amazon-670K,
    670K rótulos), onde a matriz completa `[Nq, Nl]` não cabe na memória.

    Mesma semântica EXATA do `rank_per_class` (top-`num_labels` cabeça + top-`num_labels`
    cauda, produto interno = cosine em vetores L2-normalizados), só que:
      - os embeddings de rótulo (cabeça e cauda) vão UMA vez para o `device` (GPU);
      - as queries são processadas em blocos de `chunk_size` → a matriz parcial é só
        `[chunk_size, Nl]`, não `[Nq, Nl]`.
    Resultado idêntico ao caminho numpy (mesma seleção top-k) — é um trade de memória,
    não de fidelidade. Roda em GPU (matmul de 670K rótulos é inviável em CPU numpy);
    `device="cpu"` funciona para teste com torch.
    """
    import torch

    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    label_ids_arr = np.asarray(label_ids)
    head_pos = np.array([i for i, c in enumerate(label_ids) if c in head], dtype=np.intp)
    tail_pos = np.array([i for i, c in enumerate(label_ids) if c in tail], dtype=np.intp)

    # rótulos de cada classe no device uma vez (cabeça + cauda), com as colunas alinhadas
    groups = []
    for pos in (head_pos, tail_pos):
        if pos.shape[0] == 0:
            continue
        L = torch.from_numpy(np.ascontiguousarray(label_rpr[pos])).to(dev)
        groups.append((L, label_ids_arr[pos]))

    runs: dict[str, list[tuple[int, float]]] = {}
    with torch.no_grad():
        for start in range(0, len(query_ids), chunk_size):
            block = text_rpr[start:start + chunk_size]
            q = torch.from_numpy(np.ascontiguousarray(block)).to(dev)   # [Bq, D]
            items_block: list[list[tuple[int, float]]] = [[] for _ in range(q.shape[0])]
            for L, cols in groups:
                sims = q @ L.T                                          # [Bq, Nsub]
                k = min(num_labels, L.shape[0])
                vals, idx = torch.topk(sims, k, dim=1)                  # já ordenado desc
                vals_np = vals.float().cpu().numpy()
                idx_np = idx.cpu().numpy()
                for bi in range(q.shape[0]):
                    sel = cols[idx_np[bi]]
                    items_block[bi].extend(
                        (int(c), float(s)) for c, s in zip(sel, vals_np[bi])
                    )
            for bi in range(q.shape[0]):
                runs[query_ids[start + bi]] = items_block[bi]
    return runs


def load_fold_descriptions(cfg: DenseConfig, fold_id: int) -> dict[int, str]:
    """Lê as RAG-labels do fold (se `label_enhancement="LLM"`). Ausência do arquivo
    NÃO é erro: devolve {} → resolve_label_texts cai no nome cru (opcionalidade)."""
    if cfg.label_enhancement != "LLM":
        return {}
    path = cfg.fold_rag_path(fold_id)
    desc = load_descriptions(path)
    if not desc:
        print(f"  [aviso] sem RAG-labels em {path} → usando nome cru do rótulo (fallback)")
    return desc


# ─────────────────────── encoder + loss (imports lazy) ─────────────────────────────

def build_encoder(cfg: DenseConfig):
    """Constrói o BertEncoder (BERT + ConcatenatePooling 3072-d, L2-normalizado).
    Import lazy de torch/transformers — espelha build_index do esparso."""
    import torch
    import torch.nn.functional as F
    from torch import nn
    from transformers import BertModel

    class BertEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = BertModel.from_pretrained(cfg.architecture, output_hidden_states=True)

        def forward(self, input_ids):
            attention_mask = (input_ids > 0).long()          # pad_token_id do BERT = 0
            out = self.encoder(input_ids, attention_mask)
            hs = out.hidden_states
            cat = torch.cat((hs[-1], hs[-2], hs[-3], hs[-4]), dim=-1)
            return F.normalize(cat[:, 0], p=2, dim=1)          # token [CLS], normalizado

    return BertEncoder()


_PML_CACHE: dict = {}


def _pml_classes():
    """Define (uma vez) a distância dot-product e o RelevanceMiner do RAG-Fuse.
    Import lazy de pytorch-metric-learning — só quem treina/avalia a loss precisa."""
    if _PML_CACHE:
        return _PML_CACHE["distance"], _PML_CACHE["miner"]
    import torch
    from pytorch_metric_learning.distances import BaseDistance
    from pytorch_metric_learning.miners import BaseMiner
    from pytorch_metric_learning.utils import common_functions as c_f

    c_f.check_shapes = lambda x, y: None   # RAG-Fuse desliga a checagem de shapes

    class DotProductDistance(BaseDistance):
        def __init__(self, **kwargs):
            super().__init__(is_inverted=True, **kwargs)
            assert self.is_inverted

        def check_shapes(self, query_emb, ref_emb):
            pass

        def compute_mat(self, text_rpr, label_rpr):
            return 20 * torch.einsum("ab,cb->ac", text_rpr, label_rpr)

        def pairwise_distance(self, query_emb, ref_emb):
            raise NotImplementedError

    class RelevanceMiner(BaseMiner):
        """Marca pares (text_i, label_j) do batch como positivos se label_j é gold
        do text_i (via relevance_map), senão negativos. Porte direto do RAG-Fuse."""

        def __init__(self, relevance_map: dict[int, set[int]]):
            super().__init__()
            self.relevance_map = relevance_map

        def mine(self, text_ids, label_ids):
            a1, p, a2, n = [], [], [], []
            for i, text_idx in enumerate(text_ids.tolist()):
                gold = self.relevance_map.get(text_idx, ())
                for j, label_idx in enumerate(label_ids.tolist()):
                    if label_idx >= 0:
                        if label_idx in gold:
                            a1.append(i); p.append(j)
                        else:
                            a2.append(i); n.append(j)
            dev = text_ids.device
            return (
                torch.tensor(a1, device=dev), torch.tensor(p, device=dev),
                torch.tensor(a2, device=dev), torch.tensor(n, device=dev),
            )

        def output_assertion(self, output):
            pass

    _PML_CACHE["distance"] = DotProductDistance
    _PML_CACHE["miner"] = RelevanceMiner
    return DotProductDistance, RelevanceMiner


def build_miner(relevance_map: dict[int, set[int]]):
    """RelevanceMiner instanciado (import lazy de pml). Exposto p/ teste direto."""
    _, RelevanceMiner = _pml_classes()
    return RelevanceMiner(relevance_map)


def build_loss(relevance_map: dict[int, set[int]], temperature: float = 0.07):
    """Loss NT-Xent + RelevanceMiner, fiel ao RetrieverLoss do RAG-Fuse.
    Assinatura de forward: (text_idx, text_rpr, label_idx, label_rpr) → escalar."""
    import torch
    from pytorch_metric_learning import losses

    DotProductDistance, _ = _pml_classes()
    miner = build_miner(relevance_map)
    criterion = losses.NTXentLoss(temperature=temperature, distance=DotProductDistance())

    class RetrieverLoss(torch.nn.Module):
        def forward(self, text_idx, text_rpr, label_idx, label_rpr):
            miner_outs = miner.mine(text_ids=text_idx, label_ids=label_idx)
            return criterion(text_rpr, None, miner_outs, label_rpr, None)

    return RetrieverLoss()


class FakeEncoder:
    """Dublê de encoder para testes na CPU (sem torch/transformers): embeda cada
    texto num vetor determinístico (hash → RNG) L2-normalizado. Análogo ao FakeLLM
    (label_desc) e FakeSR (retrieve_sparse). Permite exercitar rank_per_class e o
    fluxo de inferência sem tocar a GPU."""

    def __init__(self, dim: int = 16) -> None:
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.empty((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            seed = abs(hash(t)) % (2**32)
            v = np.random.RandomState(seed).randn(self.dim).astype(np.float32)
            out[i] = v / (np.linalg.norm(v) + 1e-12)
        return out


# ─────────────────────────── treino + inferência (GPU) ────────────────────────────

def _make_tokenizer(cfg: DenseConfig):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(cfg.architecture)


def _encode(encoder, tokenizer, texts: list[str], max_length: int, cfg: DenseConfig) -> np.ndarray:
    """Embeda uma lista de textos em lotes (no-grad, autocast em GPU) → [N, 3072]
    numpy L2-normalizado. Usado para queries e rótulos na inferência."""
    import torch

    device = next(encoder.parameters()).device
    use_amp = cfg.precision == "fp16" and device.type == "cuda"
    encoder.eval()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), cfg.encode_batch_size):
            batch = texts[start:start + cfg.encode_batch_size]
            ids = tokenizer(
                batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
            )["input_ids"].to(device)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                rpr = encoder(ids)
            chunks.append(rpr.float().cpu().numpy())
    return np.concatenate(chunks, axis=0) if chunks else np.empty((0, 1), dtype=np.float32)


def train_fold(
    fold: Fold,
    pooled: PooledData,
    label_texts: list[str],
    relevance_map: dict[int, set[int]],
    cfg: DenseConfig,
):
    """Treina o bi-encoder no corpus de treino do fold (contrastivo NT-Xent).
    Retorna o encoder treinado. Roda em GPU (Brev); fp16 só com CUDA."""
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import get_linear_schedule_with_warmup

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    tokenizer = _make_tokenizer(cfg)
    encoder = build_encoder(cfg).to(device)
    loss_fn = build_loss(relevance_map, cfg.temperature)

    pairs = build_train_pairs(fold, pooled)

    class PairDataset(Dataset):
        def __len__(self):
            return len(pairs)

        def __getitem__(self, i):
            text_idx, label_idx = pairs[i]
            return text_idx, label_idx

    def collate(batch):
        text_idx = [b[0] for b in batch]
        label_idx = [b[1] for b in batch]
        text_ids = tokenizer(
            [pooled.texts[t] for t in text_idx],
            padding=True, truncation=True, max_length=cfg.text_max_length, return_tensors="pt",
        )["input_ids"]
        label_ids = tokenizer(
            [label_texts[c] for c in label_idx],
            padding=True, truncation=True, max_length=cfg.label_max_length, return_tensors="pt",
        )["input_ids"]
        return (
            torch.tensor(text_idx), text_ids,
            torch.tensor(label_idx), label_ids,
        )

    loader = DataLoader(
        PairDataset(), batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, collate_fn=collate,
    )

    optimizer = torch.optim.AdamW(
        encoder.parameters(), lr=cfg.lr, betas=(0.9, 0.999), eps=1e-8,
        weight_decay=cfg.weight_decay, amsgrad=True,
    )
    total_steps = max(1, cfg.epochs * len(loader))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=round(cfg.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )
    use_amp = cfg.precision == "fp16" and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    encoder.train()
    for epoch in range(cfg.epochs):
        running = 0.0
        for text_idx, text_ids, label_idx, label_ids in loader:
            text_ids, label_ids = text_ids.to(device), label_ids.to(device)
            text_idx, label_idx = text_idx.to(device), label_idx.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", enabled=use_amp):
                text_rpr = encoder(text_ids)
                label_rpr = encoder(label_ids)
                loss = loss_fn(text_idx, text_rpr, label_idx, label_rpr)
            scaler.scale(loss).backward()
            prev_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # só avança o scheduler se o otimizador de fato passou (o GradScaler pula
            # o step quando detecta inf/nan e reduz a escala — chamar scheduler.step()
            # nesse caso desalinha o LR; ver UserWarning do PyTorch).
            if not use_amp or scaler.get_scale() >= prev_scale:
                scheduler.step()
            running += float(loss.detach())
        print(f"  época {epoch + 1}/{cfg.epochs}: loss média = {running / max(1, len(loader)):.4f}")

    return encoder, tokenizer


def infer_fold(
    encoder,
    tokenizer,
    fold: Fold,
    pooled: PooledData,
    label_texts: list[str],
    head: set[int],
    tail: set[int],
    cfg: DenseConfig,
) -> dict[str, list[tuple[int, float]]]:
    """Embeda as queries do fold + todos os rótulos e ranqueia por cosine exato
    (top-`num_labels` cabeça + cauda). Retorna {qid_global: [(col, score), ...]}."""
    query_idx = fold.test_idx.tolist()
    query_texts = [pooled.texts[i] for i in query_idx]
    query_ids = [str(i) for i in query_idx]

    text_rpr = _encode(encoder, tokenizer, query_texts, cfg.text_max_length, cfg)
    label_rpr = _encode(encoder, tokenizer, label_texts, cfg.label_max_length, cfg)
    label_ids = list(range(len(label_texts)))
    if len(label_texts) > cfg.infer_chunk_threshold:
        print(f"  inferência chunkada: {len(label_texts)} rótulos > "
              f"{cfg.infer_chunk_threshold} → blocos de {cfg.infer_chunk_size} queries no device")
        return rank_per_class_chunked(
            text_rpr, query_ids, label_rpr, label_ids, head, tail,
            cfg.num_labels, chunk_size=cfg.infer_chunk_size, device=cfg.device,
        )
    return rank_per_class(text_rpr, query_ids, label_rpr, label_ids, head, tail, cfg.num_labels)


def run_cv(cfg: DenseConfig | None = None, only_fold: int | None = None) -> None:
    """Protocolo oficial: 5-fold CV sobre o dataset agrupado. Por fold: resolve o
    texto dos rótulos (RAG-labels do fold se LLM), treina o bi-encoder no corpus do
    fold, infere nas queries do fold e escreve runs/dense.fold{f}.trec.

    Resume-safe: com cfg.resume, pula fold cujo .trec já existe (o run só é gravado
    ao FIM do fold, então um fold interrompido é refeito por inteiro — sem meio-termo
    corrompido). `only_fold` roda apenas aquele fold (resume direcionado / paralelizar
    folds em GPUs distintas)."""
    cfg = cfg or DenseConfig()
    pooled = load_pooled(cfg.raw_dir)
    label_vocab = _read_lines(os.path.join(cfg.raw_dir, "Y.txt"))
    if len(label_vocab) != pooled.n_labels:
        raise ValueError(
            f"Y.txt tem {len(label_vocab)} rótulos mas Y tem {pooled.n_labels} colunas."
        )

    head, tail = head_tail_split(pooled.label_cols, pooled.n_labels, cfg.head_frac)
    relevance_map = build_relevance_map(pooled.label_cols)
    print(
        f"agrupado: {len(pooled)} docs | rótulos: {pooled.n_labels} "
        f"| cabeça: {len(head)} | cauda: {len(tail)} (global, Pareto) "
        f"| {cfg.n_folds}-fold (seed={cfg.seed}) | enhancement: {cfg.label_enhancement} "
        f"| modelo: {cfg.architecture} | device: {cfg.device}"
        + (f" | só fold {only_fold}" if only_fold is not None else "")
    )
    # logar o config de treino: o log passa a auto-documentar batch/épocas/lr (uma
    # mistura de batch entre folds já mascarou um confound do CV — ver git log).
    print(
        f"treino: batch={cfg.batch_size} | épocas={cfg.epochs} | lr={cfg.lr} "
        f"| wd={cfg.weight_decay} | temp={cfg.temperature} | precision={cfg.precision} "
        f"| text_max={cfg.text_max_length} | label_max={cfg.label_max_length}"
    )

    folds = make_folds(len(pooled), k=cfg.n_folds, seed=cfg.seed)
    for fold in folds:
        if only_fold is not None and fold.fold_id != only_fold:
            continue
        out_path = cfg.fold_out_path(fold.fold_id)
        if cfg.resume and os.path.exists(out_path):
            print(f"\n── fold {fold.fold_id}: {out_path} já existe — pulando "
                  f"(resume; use --no-resume p/ refazer)")
            continue

        print(f"\n── fold {fold.fold_id}: corpus {len(fold.train_idx)} | queries {len(fold.test_idx)} ──")
        descriptions = load_fold_descriptions(cfg, fold.fold_id)
        label_texts = resolve_label_texts(label_vocab, descriptions, cfg.label_enhancement)

        encoder, tokenizer = train_fold(fold, pooled, label_texts, relevance_map, cfg)
        runs = infer_fold(encoder, tokenizer, fold, pooled, label_texts, head, tail, cfg)

        write_trec(runs, out_path, cfg.tag)
        print(f"run TREC salvo em {out_path} ({len(runs)} queries)")


def main(cfg: DenseConfig | None = None) -> None:
    import argparse

    from src.data import add_dataset_arg, apply_dataset

    parser = argparse.ArgumentParser(description="Recuperador denso (bi-encoder, 5-fold CV)")
    add_dataset_arg(parser)
    parser.add_argument("--fold", type=int, default=None,
                        help="roda só este fold (resume direcionado / paralelizar GPUs)")
    parser.add_argument("--device", type=str, default=None,
                        help="override do device (ex.: cuda:1)")
    parser.add_argument("--label-enhancement", type=str, default=None,
                        choices=["LLM", "NONE"],
                        help="LLM = descrições RAG-labels; NONE = só o nome cru do rótulo")
    parser.add_argument("--no-resume", action="store_true",
                        help="refaz folds mesmo que o .trec já exista")
    args, _ = parser.parse_known_args()

    cfg = cfg or DenseConfig()
    apply_dataset(cfg, args.dataset)
    if args.device:
        cfg.device = args.device
    if args.label_enhancement:
        cfg.label_enhancement = args.label_enhancement
    if args.no_resume:
        cfg.resume = False
    run_cv(cfg, only_fold=args.fold)


if __name__ == "__main__":
    main()
