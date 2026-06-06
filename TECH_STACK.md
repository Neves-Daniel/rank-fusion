# TECH_STACK — rank-fusion

Linguagem: **Python 3.11.9** (pyenv).

## Já instalado e verificado
- numpy 2.4.6
- scipy 1.17.1   (leitura das matrizes esparsas `.npz`)
- gdown 6.1.0    (usado na tentativa de download via Drive; fonte final foi HuggingFace via curl)
- **retriv 0.2.3**   (recuperador esparso BM25 — `src/retrieve_sparse.py`, validado no Eurlex-4K)
- **ranx 0.3.21**    (fusão + normalização + métricas clássicas; veio junto com o retriv)
- torch 2.12 / transformers 5.10 / nltk 3.9   (dependências do retriv)

### Setup obrigatório do NLTK (o retriv usa para tokenizar)
```
python -m nltk.downloader punkt punkt_tab stopwords
```
Sem isso, o `retriv` falha com `LookupError: Resource 'punkt_tab' not found`.

### Decisão: retriv em vez de bm25s
O TECH_STACK original previa `bm25s`. Trocamos por **retriv** para replicar fielmente
o `SparseRetriever` do RAG-Fuse (celsofranssa/RAG-Fuse): mesma tokenização
(word/stemmer/stopwords + normalizações) e BM25 com k1=1.5, b=0.75.

**Gotcha do retriv (resolvido):** queries muito longas (docs grandes da EUR-Lex)
estouram a pilha do kernel numba do retriv (segfault no `bsearch`). Não há env
var nem batch size que contorne — roda em worker thread de pilha pequena. Solução:
deduplicar os termos da query (`dedup_query_terms`, em `retrieve_sparse.py`).

## A instalar
- pyxclib (git: kunaldahiya/pyxclib)   (métricas XMTC: PSP@k, PSnDCG@k)
- tqdm

## Recuperador denso — metodologia ainda NÃO definida
Modelo, biblioteca de embedding e mecanismo de busca serão escolhidos quando a
metodologia do recuperador denso for decidida. Nada de denso instalado/fixado ainda.

**Decisão sobre o texto (2026-06-06):** o denso usa o MESMO texto stemizado do
esparso. Verificamos que todos os espelhos do EURLex-4K (thekop79, PECOS/xmc-base,
AttentionXML) trazem o texto stemizado/sem-stopwords — não há versão de palavras
inteiras com o mesmo split/rótulos. É fiel ao baseline (xCoRetriev/França usam essa
base do AttentionXML); a degradação de embeddings fica como ameaça à validade. Ver
REQUIREMENTS.md → "Casos de borda".

## Não usar
- Frameworks de classificação XMTC end-to-end (AttentionXML, LightXML) — o foco é fusão de rankings, não treinar classificador.
