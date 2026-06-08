# REQUIREMENTS — rank-fusion

## Pergunta de pesquisa
Em XMTC, quais combinações entre **algoritmos de fusão de rankings** e
**estratégias de normalização** produzem os melhores resultados na recuperação de
**tail labels**, sem comprometer o desempenho em head labels?

## Protocolo de avaliação
**5-fold cross-validation sobre o dataset agrupado** (treino+teste), fiel ao
artigo-base (não o split fixo do PECOS). Folds seeded/reprodutíveis em `splits.py`;
cabeça/cauda definida globalmente (Pareto 80/20). Ver ARCHITECTURE.md.

## Fluxo principal
1. Carregar dataset (texto + rótulos por documento) e gerar os folds de CV.
2. Gerar dois rankings base de rótulos por doc de teste:
   - esparso (BM25),
   - denso (bi-encoder BERT *fine-tuned*, label-as-document, com RAG-labels).
   Estratégia denso decidida: BERT fine-tuned (perda NT-Xent) embeda doc e rótulo
   num espaço compartilhado e ranqueia rótulos por similaridade direta doc×rótulo;
   cada rótulo é representado pela sua descrição RAG-labels (enriquecida por LLM).
3. Para cada par (normalização, algoritmo de fusão): fundir os dois rankings.
4. Avaliar com métricas globais e **de cauda**.
5. Selecionar as melhores combinações via grid search nos datasets.

## Métricas (obrigatório)
- Globais: Precision@k, nDCG@k, Recall@k (k ∈ {1,5,10}).
- **Cauda (obrigatórias): PSP@k e PSnDCG@k** (propensity-scored) e análise por
  split cabeça/cauda (Pareto: 80% dos rótulos menos frequentes = cauda).
- Nota: o artigo-base reporta P@k/nDCG@k **segmentados cabeça/cauda** (não PSP) —
  implementar esses para comparar com as tabelas dele; PSP é extensão nossa.
- Eficiência: custo computacional da fusão (tempo).
- Robustez: estabilidade das combinações entre datasets e sob ruído.

## Datasets (nesta ordem)
1. **Eurlex-4K** — validar pipeline ponta a ponta. (dados já baixados/validados)
2. Wiki10-31K.
3. AmazonCat-13K — stretch.
4. Amazon-670K — stretch (maior risco de prazo/compute).

## Casos de borda / decisões fixas
- Texto stemizado: TODOS os espelhos do EURLex-4K (thekop79, PECOS/xmc-base,
  AttentionXML) distribuem o texto stemizado e sem stopwords — é o pré-processamento
  original (Loza Mencía & Fürnkranz). NÃO existe versão "drop-in" com palavras
  inteiras e o mesmo split/rótulos. Decisão (2026-06-06): o recuperador denso usa o
  MESMO texto stemizado. Duas justificativas: (1) é fiel ao pipeline de referência —
  os artigos-base (xCoRetriev / França et al.) usam exatamente essa base do
  AttentionXML, então mantemos comparabilidade com o baseline; (2) a degradação de
  embeddings por stemização fica registrada como ameaça à validade, mas não invalida
  o estudo comparativo de fusão (ambos os recuperadores operam sobre o mesmo corpus).
  Texto cru completo só existe em outro dataset (EURLEX57K, Chalkidis 2019), com
  split e rótulos diferentes → fora de escopo, quebraria a comparabilidade.
- Rótulos nunca vistos no teste: tratados normalmente (recuperação pode não os
  alcançar; isso é esperado e medido pelas métricas).
- Texto por dataset NÃO é homogêneo: o Eurlex-4K vem stemizado/sem-stopwords; o
  Wiki10-31K (PECOS xmc-base) vem em PALAVRAS INTEIRAS e com documentos bem mais
  longos (artigos da Wikipédia). Consequência: a justificativa "denso usa texto
  stemizado por fidelidade ao baseline" é específica do Eurlex; no Wiki10 o denso
  recebe texto melhor (positivo), mas o truncamento a 512 wordpieces passa a descartar
  proporcionalmente mais conteúdo (ameaça à validade no Wiki10). Registrar por dataset,
  não assumir o pré-processamento de um para o outro.

## Fora de escopo (v1)
- Datasets além dos 4 listados.

## Cronograma (8 semanas — ver proposta)
Setup/dados → recuperação base → normalização → fusão → grid search → avaliação →
robustez/eficiência → relatório.
