"""Smoke test do backend vLLM REAL (carrega o modelo de verdade na GPU).

Diferente de test_label_desc.py (que usa o FakeLLM), aqui o vLLM gera de verdade.
Não é teste de qualidade — só confirma que o engine carrega, o chat template aplica
e a geração devolve uma string não-vazia.

Pulado automaticamente se o vllm não estiver instalado (suíte mínima fica verde).
Rodar só este (na Brev/GPU):  python -m pytest -m vllm
"""
import pytest

pytest.importorskip("vllm")  # pula o arquivo inteiro sem vllm instalado

from src.label_desc import LabelDescConfig, VLLMBackend, build_prompt

pytestmark = pytest.mark.vllm


def test_vllm_gera_descricao_nao_vazia():
    cfg = LabelDescConfig(max_tokens=64)
    backend = VLLMBackend(cfg)
    prompt = build_prompt("fisheries", pos_examples=[], neg_examples=[])
    out = backend.generate([prompt])
    assert len(out) == 1
    assert isinstance(out[0], str) and out[0].strip()
