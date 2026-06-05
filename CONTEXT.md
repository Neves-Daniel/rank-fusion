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
- **Insumo (mínimo esforço):** os rankings base esparso e denso. A metodologia do
  recuperador denso ainda NÃO está definida.

## Artigos-base
1. **xCoRetriev — França et al., SIGIR 2025.** "Optimizing Tail-Head Trade-off for
   XMTC with RAG-Labels and a Dynamic Two-Stage Retrieval and Fusion Pipeline."
   Reformula XMTC como recuperação em duas etapas (esparso + denso) com fusão
   dinâmica; usa RAG-labels para qualidade/ruído. Ganhos de até 48% em métricas
   propensity-scored. É o pipeline de referência do projeto.
2. **França et al., 2025 (arXiv 2507.03761).** "Ranking-based Fusion Algorithms for
   XMTC." Aprofunda o estágio de fusão: estuda normalizações (incl. ZMUV) e
   algoritmos de fusão; melhor combinação relatada = **CombMNZ + ZMUV**. É a base
   metodológica direta do nosso estudo.
3. **França et al., SBBD 2025.** "Muitas Classes Desbalanceadas? Não Classifique –
   Ranqueie! ... RAG-labels para Classificação Textual Multi-classe." **Origem do
   conceito de RAG-labels** (descrições de classe enriquecidas por LLM). Contexto
   multi-classe; no nosso projeto RAG-labels é stretch, fora da v1.

Todos do mesmo grupo (UFMG/UFSJ): França, Salles, Cunha, Rocha, Gonçalves (+ outros).

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
- **xCoRetriev (SIGIR 2025):** DOI 10.1145/3726302.3730052 —
  https://dl.acm.org/doi/10.1145/3726302.3730052  (texto completo atrás de paywall ACM)
- **Ranking-based Fusion (arXiv 2025):** https://arxiv.org/abs/2507.03761 —
  HTML aberto: https://arxiv.org/html/2507.03761
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
- ✅ Dados do Eurlex-4K baixados e validados (15.449 treino / 3.865 teste / 3.956 rótulos).
- ✅ Arquivos-âncora criados; metodologia do recuperador denso ainda em aberto.
- ✅ **Recuperador esparso implementado e validado** (`src/retrieve_sparse.py`): kNN léxico
  via retriv (BM25 k1=1.5/b=0.75), agregação de vizinhos + split cabeça/cauda → run TREC.
  Sanidade no Eurlex: query 0 recuperou 3/5 rótulos-gold no top-10. Decisões: chave de
  rótulo = `label_{coluna}`; sem filtro `text_cls` (artefato multi-classe do RAG-Fuse).
- ⬜ Recuperador denso, fusão, métricas (PSP@k) e grid search: a fazer.
- ⏳ Falta rodar o esparso completo (3.865 queries, cutoff=100) e gerar `runs/sparse.trec`.
