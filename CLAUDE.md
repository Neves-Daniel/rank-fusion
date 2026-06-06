# Guardrails do Projeto: rank-fusion (XMTC)

## Perfil
Projeto de pesquisa em PLN/Recuperação de Informação (mestrado, UFMG). Código é
**experimental e reprodutível**, não produção. Priorize clareza, reprodutibilidade
e honestidade científica sobre engenharia sofisticada.

## Objetivo (não desviar disto)
Estudar quais combinações de **algoritmos de fusão de rankings × estratégias de
normalização** melhoram a recuperação de **tail labels** em XMTC sem prejudicar
head labels.

**Referência principal:** França, Rabbi, Salles, Cunha, Rocha & Gonçalves (2025),
"Ranking-based Fusion Algorithms for XMTC", arXiv 2507.03761. Define protocolo
(5-fold CV), métricas (P@k/nDCG@k segmentados cabeça/cauda), datasets e baseline
(melhor: CombMNZ+ZMUV). Demais artigos e links em CONTEXT.md → "Artigos-base".

## Guardrails de Design (Regras de Ouro)
- **Métricas de cauda são obrigatórias.** Nunca reportar só Precision@k/Recall@k
  (dominadas pela cabeça). Sempre incluir **PSP@k / PSnDCG@k** e/ou split
  cabeça-cauda (Pareto 80%). A pergunta de pesquisa é sobre tail labels.
- **Formato canônico de ranking = TREC run** (`qid label_id rank score`). Todo
  recuperador exporta nesse formato; fusão e avaliação consomem dele.
- **Fusão/normalização/métricas clássicas via `ranx`.** Implementar 2-3 algoritmos
  à mão (CombMNZ, RRF) só para demonstrar entendimento e validar contra o ranx.
- **Reprodutibilidade:** fixar seeds; salvar configs de cada experimento; não
  hard-codear caminhos fora de `configs/`.
- **Começar pequeno:** validar tudo no Eurlex-4K antes de escalar para
  Wiki10-31K → AmazonCat-13K → Amazon-670K.
- **Honestidade:** registrar limitações (ex.: texto stemizado degrada embeddings),
  reportar falhas de teste como falhas, não maquiar resultados.

## Anti-padrões a evitar
- Rodar Amazon-670K antes de validar o pipeline no Eurlex-4K.
- Recomputar recuperação base a cada experimento de fusão (gerar runs uma vez,
  iterar fusão offline).
- Adicionar dependências pesadas sem confirmar com o usuário.

## Stack
Ver TECH_STACK.md. Arquitetura e formatos: ARCHITECTURE.md.

## Memória persistente (padrão por categoria)
Memórias ficam em subpastas de `~/.claude/projects/-home-dnpin-nlp-rank-fusion/memory/`
por categoria (`projeto/`, `usuario/`, `ambiente/`, ...). Cada pasta tem um
`README.md` como índice local; o `MEMORY.md` raiz lista só as categorias. Use a
skill `/save-memory` para salvar no local correto e o agent `memory-curator` para
revisão periódica. **Nunca** salve memórias direto na raiz do diretório de memória.
