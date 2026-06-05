# REQUIREMENTS — rank-fusion

## Pergunta de pesquisa
Em XMTC, quais combinações entre **algoritmos de fusão de rankings** e
**estratégias de normalização** produzem os melhores resultados na recuperação de
**tail labels**, sem comprometer o desempenho em head labels?

## Fluxo principal
1. Carregar dataset (texto + rótulos por documento).
2. Gerar dois rankings base de rótulos por doc de teste:
   - esparso (BM25),
   - denso (metodologia a definir).
   A estratégia que converte documento em ranking de rótulos ainda será decidida.
3. Para cada par (normalização, algoritmo de fusão): fundir os dois rankings.
4. Avaliar com métricas globais e **de cauda**.
5. Selecionar as melhores combinações via grid search nos datasets.

## Métricas (obrigatório)
- Globais: Precision@k, nDCG@k, Recall@k (k ∈ {1,5,10}).
- **Cauda (obrigatórias): PSP@k e PSnDCG@k** (propensity-scored) e análise por
  split cabeça/cauda (Pareto: 80% dos rótulos menos frequentes = cauda).
- Eficiência: custo computacional da fusão (tempo).
- Robustez: estabilidade das combinações entre datasets e sob ruído.

## Datasets (nesta ordem)
1. **Eurlex-4K** — validar pipeline ponta a ponta. (dados já baixados/validados)
2. Wiki10-31K.
3. AmazonCat-13K — stretch.
4. Amazon-670K — stretch (maior risco de prazo/compute).

## Casos de borda / decisões fixas
- Texto stemizado do espelho atual pode degradar recuperadores densos → registrar
  como ameaça à validade; não invalida o estudo comparativo de fusão.
- Rótulos nunca vistos no teste: tratados normalmente (recuperação pode não os
  alcançar; isso é esperado e medido pelas métricas).

## Fora de escopo (v1)
- Geração de RAG-labels (componente de qualidade do xCoRetriev; stretch).
- Fine-tuning de BERT / treino de classificador XMTC.
- Datasets além dos 4 listados.

## Cronograma (8 semanas — ver proposta)
Setup/dados → recuperação base → normalização → fusão → grid search → avaliação →
robustez/eficiência → relatório.
