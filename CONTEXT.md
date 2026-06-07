# CONTEXT — visão do projeto e artigos-base

Documento de referência para retomar o contexto em qualquer interação futura.
Detalhes operacionais ficam em CLAUDE.md / TECH_STACK.md / ARCHITECTURE.md / REQUIREMENTS.md.

## A ideia em uma frase
Em **XMTC** (classificação multi-rótulo com espaço enorme de rótulos e distribuição
*long-tail*), tratamos a atribuição de rótulos como **recuperação de informação**:
o documento é a *query*, os rótulos são recuperados e ranqueados. Estudamos quais
**(algoritmo de fusão de rankings × estratégia de normalização)** melhor combinam um
ranking **esparso** e um **denso** para melhorar os **tail labels** sem prejudicar os
**head labels**.

## Por que fusão
Recuperadores diferentes concordam mais nos rótulos certos do que nos errados.
Esparso (léxico) e denso (semântico) erram de formas complementares;
fundir os dois rankings reforça acertos e reduz ruído — útil justamente na cauda.

## O que é contribuição vs. insumo
- **Contribuição:** o estudo comparativo de fusão × normalização para tail labels.
- **Insumo:** os rankings base esparso e denso. O denso reproduz o do artigo
  principal: bi-encoder BERT *fine-tuned* (espaço compartilhado texto–rótulo,
  perda contrastiva NT-Xent), estratégia *label-as-document* com rótulos
  representados por **RAG-labels** (descrições enriquecidas por LLM).

## Artigos-base
1. **⭐ REFERÊNCIA PRINCIPAL — França, Rabbi, Salles, Cunha, Rocha & Gonçalves, 2025
   (arXiv 2507.03761).** "Ranking-based Fusion Algorithms for Extreme Multi-label
   Text Classification (XMTC)." É o artigo que define o nosso estudo: 6 estratégias
   de normalização × 10 algoritmos de fusão sobre rankings esparso+denso, avaliados
   por P@k/nDCG@k segmentados cabeça/cauda em 5-fold CV nos 4 datasets. Melhor
   combinação relatada = **CombMNZ + ZMUV**. Protocolo, métricas, datasets e baseline
   do projeto saem daqui (ver Tabelas 1–2 para Eurlex-4K).
2. **xCoRetriev — França et al., SIGIR 2025.** "Optimizing Tail-Head Trade-off for
   XMTC with RAG-Labels and a Dynamic Two-Stage Retrieval and Fusion Pipeline."
   Contexto/pipeline mais amplo (duas etapas esparso+denso com fusão dinâmica +
   RAG-labels; ganhos de até 48% em métricas propensity-scored). De onde vem o
   recuperador esparso (kNN léxico) e o split 64+64 cabeça/cauda.
3. **França et al., SBBD 2025.** "Muitas Classes Desbalanceadas? Não Classifique –
   Ranqueie! ... RAG-labels para Classificação Textual Multi-classe." **Origem do
   conceito de RAG-labels** (descrições de classe enriquecidas por LLM). Contexto
   multi-classe; **RAG-labels agora está em escopo** no nosso projeto: o
   recuperador denso representa cada rótulo pela sua descrição RAG-labels.

Todos do mesmo grupo (UFMG/UFSJ): França, Rabbi, Salles, Cunha, Rocha, Gonçalves.

## Glossário rápido
- **XMTC:** Extreme Multi-Label Text Classification.
- **Head / tail labels:** rótulos frequentes / raros (split Pareto: 80% menos
  frequentes = cauda).
- **Esparso / denso:** recuperador léxico (ex.: BM25) / baseado em embeddings.
- **Normalização:** tornar scores comparáveis antes de fundir (Min-Max, ZMUV, Rank...).
- **Fusão:** combinar múltiplos rankings (CombSUM, CombMNZ, RRF, ISR, Borda...).
- **PSP@k / PSnDCG@k:** métricas *propensity-scored*; medem desempenho com peso para
  a cauda — obrigatórias neste projeto.
- **RAG-labels:** descrições de rótulo enriquecidas via LLM/RAG (do artigo SBBD).

## Datasets
Eurlex-4K (validação inicial, já baixado) → Wiki10-31K → AmazonCat-13K → Amazon-670K.

## Links das referências
- **⭐ Ranking-based Fusion (arXiv 2025) — PRINCIPAL:** https://arxiv.org/abs/2507.03761 —
  HTML aberto: https://arxiv.org/html/2507.03761
- **xCoRetriev (SIGIR 2025):** DOI 10.1145/3726302.3730052 —
  https://dl.acm.org/doi/10.1145/3726302.3730052  (texto completo atrás de paywall ACM)
- **RAG-labels (SBBD 2025):** https://sol.sbc.org.br/index.php/sbbd/article/view/37243

## Acesso rápido (dados e ferramentas)
- **Dados no disco:** `data/eurlex4k/raw/` (texto + matrizes de rótulos, formato PECOS).
  Recarregar/baixar: `bash scripts/download_eurlex.sh`.
- **Carregar em Python:** `from src.data import load_dataset; ds = load_dataset("data/eurlex4k/raw")`
  (ver `python src/data.py` para um resumo/estatísticas do dataset).
- **Espelho dos dados (Eurlex-4K, com texto):** https://huggingface.co/datasets/thekop79/EURLex-4K
  (o Google Drive do AttentionXML falha no gdown). Outros datasets:
  Extreme Classification Repository — http://manikvarma.org/downloads/XC/XMLRepository.html
- **Ferramentas-chave:** retriv (BM25 esparso) https://github.com/AmenRa/retriv ·
  ranx (fusão/norm/métricas) https://github.com/AmenRa/ranx ·
  pyxclib (PSP@k) https://github.com/kunaldahiya/pyxclib
- **Código de referência (autor):** RAG-Fuse https://github.com/celsofranssa/RAG-Fuse
  (multi-classe; nosso esparso replica o `SparseRetriever` dele).
- **Memória persistente do projeto (entre sessões):**
  `~/.claude/projects/-home-dnpin-nlp-rank-fusion/memory/` (índice em `MEMORY.md`).

## Estado atual (atualizar conforme avança)
- ✅ Dados do Eurlex-4K baixados e validados (15.449 treino / 3.865 teste / 3.956 rótulos;
  N=19.314 agrupado, bate com a Tabela 1 do artigo).
- ✅ Arquivos-âncora criados; metodologia do recuperador denso definida: bi-encoder
  BERT *fine-tuned* (label-as-document) + RAG-labels (ver TECH_STACK.md). Falta implementar.
- ✅ **Recuperador esparso implementado e validado** (`src/retrieve_sparse.py`): kNN léxico
  via retriv (BM25 k1=1.5/b=0.75), agregação de vizinhos + split cabeça/cauda → run TREC.
  Sanidade no Eurlex: query 0 recuperou 3/5 rótulos-gold no top-10. Decisões: chave de
  rótulo = `label_{coluna}`; sem filtro `text_cls` (artefato multi-classe do RAG-Fuse).
- ✅ **Splits de 5-fold CV implementados** (`src/splits.py`): protocolo oficial fiel ao
  artigo (CV sobre dataset agrupado, cabeça/cauda global). `run_cv` gera 1 run por fold.
  Validado: 24 testes leves verdes + folds reais (~3.863 queries/fold; cauda = 80%).
- ✅ **RAG-labels (Etapa 1 do denso) implementadas** (`src/label_desc.py`): descrição por
  rótulo via LLM (Llama-3.1-8B/vLLM), **por fold**, fiel à task `label_desc` do RAG-Fuse.
  13 testes CPU verdes (FakeLLM); falta `pip install vllm` + rodar na Brev (gera os JSONL por fold).
- ✅ **Recuperador denso implementado** (`src/retrieve_dense.py`): porte do
  `DenseRetriever` do RAG-Fuse em torch+transformers puros — BERT + ConcatenatePooling
  (3072-d), NT-Xent (temp 0.07, via pytorch-metric-learning) + miner por relevance-map,
  AdamW lr=5e-5 + warmup linear, 5 ép./fp16, por fold; inferência por cosine exato →
  64+64 → `runs/dense.fold{f}.trec`. **RAG-labels OPCIONAL** via `label_enhancement`
  (`"LLM"` usa as descrições do fold; `"NONE"` cai no nome cru — fallback automático).
  15 testes CPU verdes (FakeEncoder; loss/encoder reais são opt-in, marker `bert`).
  Falta `pip install pytorch-metric-learning` + treinar/inferir na Brev.
- ✅ **Fusão implementada** (`src/fusion.py`): wrappers ranx para as 6 normalizações ×
  10 fusões do artigo (melhor = CombMNZ+ZMUV) + extras p/ teste (norma min-max-inverted;
  fusões rrf, logn_isr, rbc[φ], gmnz[γ]) = **7 norm × 14 fusão = 98 combinações**.
  CombMNZ e RRF reimplementados à mão e validados contra o ranx (igualdade de ordem).
  `run_cv` funde o melhor par por fold; `run_grid` funde as 98 combinações. 14 testes
  CPU verdes. Insumo do grid search. Import do ranx é lazy.
- ⬜ Métricas e grid search: a fazer.
  - Métricas escolhidas: P@k/nDCG@k **segmentados cabeça/cauda** (p/ bater com Tabela 2) +
    PSP@k/PSnDCG@k como extensão do projeto. O artigo NÃO usa PSP.
- ⏳ Falta rodar o esparso completo na Brev (`python -m src.retrieve_sparse`, 5 folds) →
  `runs/sparse.fold{0..4}.trec`.
