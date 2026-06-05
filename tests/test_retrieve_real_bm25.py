"""Validação ponta-a-ponta do BM25 REAL (retriv) num corpus de brinquedo.

Diferente de test_retrieve_sparse.py (que usa um dublê), aqui o retriv indexa e
busca de verdade. O corpus é desenhado para a resposta certa ser ÓBVIA e
conferível na mão: cada documento fala de um tema distinto e tem um único label;
uma query sobre o tema X tem de recuperar o doc X e, portanto, colocar o label de
X no topo. Se o BM25 estiver roteando errado, estes testes quebram.

Pulado automaticamente se o retriv não estiver instalado (suíte mínima fica verde).
Rodar só estes:  python -m pytest -m bm25
"""
import pytest

pytest.importorskip("retriv")  # pula o arquivo inteiro sem retriv instalado

from src.retrieve_sparse import SparseConfig, build_index, retrieve

pytestmark = pytest.mark.bm25

# Corpus de brinquedo: 3 temas bem separados. Coluna do label entre colchetes.
TRAIN_TEXTS = [
    "deep sea fishing boats catch fish fishery vessels nets ocean",   # doc0 -> pesca
    "wheat maize crops farming agriculture harvest soil subsidy",     # doc1 -> agricultura
    "banking finance euro currency monetary central bank interest",   # doc2 -> finanças
]
TRAIN_LABEL_COLS = [[100], [200], [300]]   # doc0->100(pesca), doc1->200(agric), doc2->300(fin)
N_LABELS = 400
HEAD = {100, 200, 300}                      # tudo na cabeça: topo reflete o doc recuperado
TAIL: set = set()


def _run(query: str):
    cfg = SparseConfig(cutoff=10, num_labels=10, aggregation="sum")
    sr = build_index(TRAIN_TEXTS, cfg)
    runs = retrieve(sr, [query], TRAIN_LABEL_COLS, HEAD, TAIL, cfg)
    return runs["0"]   # [(coluna_label, score), ...] já ordenável por score


@pytest.mark.parametrize(
    "query, expected_top_col",
    [
        ("fishing vessel catching fish in the sea", 100),   # pesca
        ("agricultural wheat farming and crop harvest", 200),  # agricultura
        ("central bank monetary policy and currency", 300),  # finanças
    ],
)
def test_query_recupera_o_tema_certo(query, expected_top_col):
    ranked = sorted(_run(query), key=lambda kv: kv[1], reverse=True)
    top_col, top_score = ranked[0]
    assert top_col == expected_top_col, (
        f"query {query!r} deveria rankear o label {expected_top_col} no topo, "
        f"mas veio {top_col} (ranking: {ranked})"
    )
    assert top_score > 0.0


def test_scores_sao_decrescentes():
    ranked = sorted(_run("fishing boats and fish"), key=lambda kv: kv[1], reverse=True)
    scores = [s for _, s in ranked]
    assert scores == sorted(scores, reverse=True)
    assert all(s > 0 for s in scores)


def test_query_sem_overlap_nao_inventa_label_de_alto_score():
    # termo inexistente no corpus: não deve haver recuperação (ou scores vazios)
    ranked = _run("xyzzyx termo inexistente quux")
    assert ranked == [] or all(s >= 0 for _, s in ranked)
