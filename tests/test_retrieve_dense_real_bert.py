"""Smoke test do encoder/treino REAIS (BERT via transformers).

Diferente de test_retrieve_dense.py (que usa o FakeEncoder), aqui o BERT roda de
verdade — mas num modelo PEQUENO (4 camadas) para ser rápido na CPU. Não é teste de
qualidade: confirma que o ConcatenatePooling produz 3072-d (= 4×hidden), os vetores
saem L2-normalizados, e um passo de treino contrastivo reduz a loss.

Pulado automaticamente se transformers/pytorch-metric-learning não estiverem
instalados (suíte mínima fica verde). Rodar só estes:  python -m pytest -m bert
"""
import numpy as np
import pytest

pytest.importorskip("transformers")
pytest.importorskip("pytorch_metric_learning")

import torch  # noqa: E402

from src.retrieve_dense import (  # noqa: E402
    DenseConfig,
    build_encoder,
    build_loss,
    build_relevance_map,
)

pytestmark = pytest.mark.bert

# BERT pequeno (4 camadas, hidden 256) → ConcatenatePooling = 4×256 = 1024-d.
SMALL_BERT = "google/bert_uncased_L-4_H-256_A-4"


def _cfg() -> DenseConfig:
    return DenseConfig(architecture=SMALL_BERT, device="cpu", precision="fp32")


def test_encoder_dim_e_normalizacao():
    cfg = _cfg()
    encoder = build_encoder(cfg)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(SMALL_BERT)
    ids = tok(["a fishing boat", "customs duty law"], padding=True, return_tensors="pt")["input_ids"]
    with torch.no_grad():
        rpr = encoder(ids)
    assert rpr.shape == (2, 1024)                                  # 4 camadas × hidden 256
    assert np.allclose(rpr.norm(dim=1).numpy(), 1.0, atol=1e-4)    # L2-normalizado


def test_um_passo_de_treino_reduz_a_loss():
    cfg = _cfg()
    encoder = build_encoder(cfg)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(SMALL_BERT)

    # doc0 ~ rótulo 10 (pesca); doc1 ~ rótulo 11 (finanças). relevance_map global.
    docs = ["deep sea fishing vessels and nets", "central bank monetary interest rates"]
    labels = ["fisheries fishing boats and the sea", "finance banking money and currency"]
    relevance_map = build_relevance_map([[10], [11]])
    loss_fn = build_loss(relevance_map, cfg.temperature)

    text_ids = tok(docs, padding=True, return_tensors="pt")["input_ids"]
    label_ids = tok(labels, padding=True, return_tensors="pt")["input_ids"]
    t_idx = torch.tensor([0, 1])
    l_idx = torch.tensor([10, 11])

    opt = torch.optim.AdamW(encoder.parameters(), lr=1e-3)

    def step():
        text_rpr = encoder(text_ids)
        label_rpr = encoder(label_ids)
        return loss_fn(t_idx, text_rpr, l_idx, label_rpr)

    encoder.train()
    loss0 = step()
    for _ in range(5):
        opt.zero_grad()
        loss = step()
        loss.backward()
        opt.step()
    assert step().item() < loss0.item()   # o treino diminui a loss
