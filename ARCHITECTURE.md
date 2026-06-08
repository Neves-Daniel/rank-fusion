# ARCHITECTURE — rank-fusion

## Estrutura de diretórios
```
rank-fusion/
├─ CLAUDE.md, TECH_STACK.md, ARCHITECTURE.md, REQUIREMENTS.md   # âncoras
├─ requirements.txt
├─ scripts/        # download e utilitários de linha de comando
│  └─ download_eurlex.sh
├─ src/
│  ├─ data.py            # carregar datasets (formato PECOS) -> Dataset/Split
│  ├─ splits.py          # 5-fold CV sobre dataset agrupado (treino+teste)      (feito)
│  ├─ retrieve_sparse.py # recuperador esparso (BM25 kNN, retriv) -> run TREC   (feito)
│  ├─ label_desc.py      # RAG-labels: descrição de rótulo via LLM (vLLM), por fold (feito)
│  ├─ prompts/           # templates de prompt (label_desc_prompt.txt, verbatim RAG-Fuse)
│  ├─ retrieve_dense.py  # denso: bi-encoder BERT fine-tuned (label-as-document) -> run TREC (feito)
│  ├─ fusion.py          # wrappers ranx (6 norm × 10 fusão) + CombMNZ/RRF próprios (feito)
│  ├─ metrics.py         # P@k/nDCG@k/Recall@k segmentados cabeça/cauda via ranx (feito; PSP a fazer)
│  └─ gridsearch.py      # varre 98 (fusão × norm), avalia segmentado, ranqueia por cauda + CSV (feito)
├─ configs/        # 1 arquivo por experimento (pares fusão×norm, k, etc.)
├─ data/<dataset>/
│  ├─ raw/         # arquivos baixados (NÃO versionar — ver .claudeignore/.gitignore)
│  ├─ rag-labels/  # descrições de rótulo por fold (fold{f}/labels_descriptions.jsonl)
│  └─ runs/        # run files TREC gerados pelos recuperadores
└─ notebooks/      # análise exploratória
```

## Protocolo de avaliação: 5-fold CV (fiel ao artigo)
A avaliação é **5-fold cross-validation sobre o dataset AGRUPADO** (treino+teste),
não o split fixo do PECOS — é o que o artigo-base (França et al. 2025) usa
("averaged across the five test splits"). `splits.py` junta os N docs na ordem
global `[treino, teste]`, gera k folds seeded (reprodutíveis) e expõe, por fold,
o corpus (4/5) e as queries (1/5). Cabeça/cauda (Pareto 80/20) é **global** —
frequências sobre os N docs, como o artigo define `T` —, não por fold. A seed dos
folds do artigo é desconhecida: reproduzimos a METODOLOGIA, não os índices exatos.

## Fluxo do pipeline (estágios desacoplados por arquivos)
1. **data.py** lê `raw/` → objetos em memória (texto + rótulos por doc).
   **splits.py** agrupa treino+teste e gera os folds de CV.
1.5 **label_desc.py** gera, **por fold**, uma descrição por rótulo (RAG-labels) com um
   LLM (Llama-3.1-8B via vLLM), usando só o corpus de treino do fold (sem vazamento).
   Saída: `rag-labels/fold{f}/labels_descriptions.jsonl`. Insumo do lado-rótulo do denso.
2. **retrieve_sparse.py / retrieve_dense.py** geram, para cada doc de teste (query)
   de cada fold, um ranking de rótulos. Saída = `runs/{sparse,dense}.fold{f}.trec`,
   onde `qid` = índice GLOBAL do doc agrupado (o gold é `label_cols[qid]`).
   - **Esparso (definido):** kNN léxico, à la xCoRetriev/RAG-Fuse. Indexa os docs de
     TREINO com BM25 (retriv, k1=1.5, b=0.75); para cada doc de teste recupera os
     top-`cutoff` vizinhos e agrega (`sum`/`max`) os rótulos-gold deles ponderados
     pelo score BM25. Split cabeça/cauda (Pareto) → `num_labels` por classe
     (64+64=128 no artigo). Sem treino: BM25 é estatístico.
     - *Dedup de query (obrigatório):* os termos da query são deduplicados antes
       da busca. Docs longos da EUR-Lex (até ~59k tokens, ex. o Código Aduaneiro)
       estouravam a pilha do kernel numba do retriv (lista de postings não
       deduplicada → recursão/aliasing → segfault). Dedup resolve; custo: pesos
       de query binários (ignora a frequência do termo na query). Os docs são
       legítimos — não é defeito de dados.
   - **Denso (implementado, `retrieve_dense.py`):** bi-encoder BERT *fine-tuned*
     (porte do `DenseRetriever` do RAG-Fuse em torch+transformers puros). Encoder =
     `bert-base-uncased` + ConcatenatePooling (concatena as 4 últimas camadas no
     [CLS] → 3072-d, L2-normalizado). Treino contrastivo por fold: perda NT-Xent
     (temp 0.07, via `pytorch-metric-learning`) + miner por relevance-map; AdamW
     lr=5e-5 + warmup linear, 5 épocas, fp16. Inferência: embeda doc e rótulo no
     espaço compartilhado e ranqueia por **similaridade cosine exata** (matmul; sem
     HNSW/nmslib — só ~4k rótulos), mantendo top-64 cabeça + top-64 cauda = 128.
     - *RAG-labels OPCIONAL (knob `label_enhancement`):* `"LLM"` representa cada
       rótulo por `f"{nome} {descrição RAG-labels do fold}"` (estágio 1.5); `"NONE"`
       usa só o nome cru do EuroVoc. Fallback automático ao nome cru se o JSONL do
       fold não existir. É o mecanismo nativo do RAG-Fuse, não uma gambiarra.
3. **fusion.py** combina os runs (normalização + algoritmo de fusão), por fold → run fundido.
4. **metrics.py** avalia cada run contra o gold (qrels = `label_cols[qid]`), por fold;
   reporta média ± desvio dos folds. Sempre inclui métricas de cauda.
5. **gridsearch.py** varre o produto (fusão × normalização) reusando os runs base
   já salvos — fusão é offline e barata.

## Convenções de formato
- **Run TREC:** `qid  Q0  label_id  rank  score  tag` (ou objeto `ranx.Run`).
- **Qrels:** rótulos verdadeiros do doc de teste (relevância binária).
- **IDs de rótulo nos runs:** chave `label_{índice_da_coluna}` (como no RAG-Fuse).
  Decisão revista: os rótulos textuais de `Y.txt` contêm espaços (ex.: "abolition of
  customs duties") e quebrariam o formato TREC; o índice de coluna é estável dentro
  de um dataset. O texto correspondente fica em `Y.txt` (coluna i = linha i) e serve
  para interpretação/depuração, não como chave. (A comparabilidade entre datasets é
  garantida na avaliação, que é por dataset, não pela chave compartilhada.)
- Cada estágio lê/escreve arquivos; nada de estado global escondido.

## Dados (formato PECOS — espelho thekop79/EURLex-4K)
- `trn_X.txt`/`tst_X.txt`: 1 doc/linha (texto pré-processado/stemizado).
- `Y.trn.npz`/`Y.tst.npz`: matriz esparsa CSR docs×rótulos (binária).
- `Y.txt`: vocabulário; linha i = rótulo da coluna i.
