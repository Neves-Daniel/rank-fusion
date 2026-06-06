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
- vllm (SÓ na Brev/GPU)   (serve Llama-3.1-8B-Instruct p/ gerar as RAG-labels; ver src/label_desc.py)
- pytorch-metric-learning (>=2.0)   (loss NT-Xent + miner do denso; ver src/retrieve_dense.py)
  - torch/transformers já vêm via retriv; NÃO usamos sentence-transformers/faiss
    (encoder e similaridade doc×rótulo são feitos à mão — exatos e reprodutíveis).

## Recuperador denso — bi-encoder BERT fine-tuned (label-as-document) — IMPLEMENTADO
`src/retrieve_dense.py`: porte fiel do `DenseRetriever` do RAG-Fuse em
**torch+transformers puros** (sem PyTorch-Lightning/Hydra/nmslib), no estilo do
porte do esparso (retriv).
- **Encoder:** `bert-base-uncased` (output_hidden_states) + ConcatenatePooling
  (concatena as 4 últimas camadas no token [CLS] → **3072-d**, L2-normalizado).
- **Loss/treino:** NT-Xent (temp 0.07) + miner por relevance-map, via
  `pytorch-metric-learning` (`losses.NTXentLoss` + `DotProductDistance` escala 20 +
  porte do `RelevanceMiner`). Otimização = **AdamW lr=5e-5, wd=1e-2, amsgrad +
  warmup linear** (config ATIVA do RAG-Fuse; o CyclicLR de lá está comentado —
  fica como knob alternativo). 5 épocas, fp16, **por fold** (só o corpus de treino
  do fold → sem vazamento).
- **Inferência:** embeda doc e rótulo no espaço compartilhado e ranqueia por
  **similaridade cosine EXATA** (matmul; troquei o HNSW/nmslib do RAG-Fuse — só
  ~4k rótulos, exato é mais simples/reprodutível). Mantém 64 cabeça + 64 cauda =
  128 candidatos por query → run TREC (`runs/dense.fold{f}.trec`).
- **RAG-labels OPCIONAL** via `label_enhancement` (knob nativo do RAG-Fuse):
  `"LLM"` = `f"{nome} {descrição RAG-labels do fold}"`; `"NONE"` = só o nome cru.
  Fallback automático ao nome cru se o JSONL do fold não existir.
- **Imports lazy** (torch/transformers/pml só dentro de build_encoder/build_loss):
  importar o módulo e rodar a suíte mínima (FakeEncoder) não toca a stack de GPU;
  testes da loss/encoder reais são opt-in (marker `bert`). Treino real só na Brev.
- Pendência de borda mantida: docs longos da EUR-Lex truncados a 512 wordpieces
  (`text_max_length`); descrição de rótulo a `label_max_length=256`.
**Decisão (2026-06-06):** fine-tuning e RAG-labels entraram no escopo.

**RAG-labels — geração (implementado 2026-06-06):** `src/label_desc.py` replica a task
`label_desc` do RAG-Fuse — para cada rótulo, um LLM escreve uma descrição a partir de
exemplos contrastivos (docs com/sem o rótulo). LLM = **Llama-3.1-8B-Instruct via vLLM**
(local na Brev; o RAG-Fuse usava AWS Bedrock — só o cliente muda). Geradas **por fold**
(só o corpus de treino do fold → evita vazamento test→descrição) e cacheadas em
`data/<ds>/rag-labels/fold{f}/labels_descriptions.jsonl` (append-only, resume idempotente).
Prompt verbatim em `src/prompts/label_desc_prompt.txt`. Import do vLLM é lazy: a suíte
mínima de testes roda na CPU com um dublê (FakeLLM); o smoke real é opt-in (`-m vllm`).

**Decisão sobre o texto (2026-06-06):** o denso usa o MESMO texto stemizado do
esparso. Verificamos que todos os espelhos do EURLex-4K (thekop79, PECOS/xmc-base,
AttentionXML) trazem o texto stemizado/sem-stopwords — não há versão de palavras
inteiras com o mesmo split/rótulos. É fiel ao baseline (xCoRetriev/França usam essa
base do AttentionXML); a degradação de embeddings fica como ameaça à validade. Ver
REQUIREMENTS.md → "Casos de borda".

## Não usar
- Frameworks de classificação XMTC end-to-end (AttentionXML, LightXML) — o foco é fusão de rankings, não treinar classificador.
