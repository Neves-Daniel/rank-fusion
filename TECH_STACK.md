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

## Recuperador denso — bi-encoder BERT fine-tuned (label-as-document)
Reproduz o denso do artigo principal: um BERT *fine-tuned* mapeia documento e
rótulo num espaço vetorial compartilhado (estratégia *label-as-document*),
treinado com perda contrastiva **NT-Xent** (InfoNCE); LR cíclico (~5e-5–5e-3),
~3 épocas. Cada rótulo é representado pela sua descrição **RAG-labels**
(enriquecida por LLM), não pelo nome cru do EuroVoc. Inferência: score doc×rótulo
(cosine/dot), 64 cabeça + 64 cauda = 128 candidatos por query → run TREC.
Biblioteca/treino a fixar no plano de implementação (sentence-transformers vs
loop próprio HuggingFace+torch); deps densas em `requirements.txt` ainda
provisórias. **Decisão (2026-06-06):** fine-tuning e RAG-labels deixaram de ser
guardrail/stretch e entraram no escopo.

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
