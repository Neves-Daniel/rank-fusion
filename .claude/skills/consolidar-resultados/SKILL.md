---
name: consolidar-resultados
description: Consolida os results/gridsearch.csv dos datasets do rank-fusion em tabelas no formato do artigo de referência (pares normalização × fusão, segmentadas cabeça/cauda, à la Tabelas 2–5 de França et al. 2025). Valida completude (quais folds/combos/datasets existem) e marca explicitamente o que está PENDENTE. Aciona ao montar/atualizar tabelas de resultados, comparar combinações ou checar o que já foi medido.
---

# Consolidar resultados — rank-fusion (XMTC)

Você transforma os CSVs de grid search em **tabelas prontas para o paper** e num
**relatório de completude**. É uma tarefa mecânica e determinística: você lê
números que já existem, agrega e formata — **não** roda experimento, **não**
interpreta ("melhor"/"supera") e **não** inventa célula faltante (vira placeholder
explícito). É o passo que alimenta a seção Results e o `verificador-numerico`.

## A grade é ABERTA (não cravar a lista de fusões/normalizações)
O objetivo do projeto é **acrescentar mais técnicas de fusão e normalização** ao
longo do tempo. Portanto: **descubra os valores reais a partir do CSV a cada
execução** — nunca embuta uma lista fixa de métodos/normalizações, nem um total fixo
de combinações, nem nas instruções nem nas tabelas. As colunas/linhas das tabelas e
todas as contagens saem **dos dados presentes**, não de um número memorizado. Quando
o Daniel adicionar uma fusão nova, esta skill deve passar a incluí-la sem edição.

## Fonte de dados (única verdade)
- **`data/<dataset>/results/gridsearch.csv`** — long-format, gerado por
  `src/gridsearch.py::save_csv`. Schema **exato** (header):
  `method,norm,segment,metric,mean,std`.
  - `method` — algoritmo de fusão (conjunto **aberto**; descubra com
    `cut -d, -f1 | sort -u`).
  - `norm` — normalização (conjunto **aberto**; descubra com `cut -d, -f2 | sort -u`).
  - `segment` — `overall`, `head`, `tail`. **As três sempre** (cauda obrigatória).
  - `metric` — família `precision@k`/`ndcg@k`/`recall@k` (descubra os k presentes
    com `cut -d, -f4 | sort -u`; PSP@k/PSnDCG@k entram aqui quando forem
    implementados).
  - `mean`, `std` — **média ± desvio entre os folds** (já agregado).
- Datasets possíveis: `eurlex4k`, `wiki10-31k`, `amazoncat-13k`, `amazon-670k` (a
  lista também é aberta). O backup espelho fica em
  `~/nlp/brev-backups/data/<dataset>/results/` — se o CSV não estiver em `data/`,
  **procure no backup** e diga de onde veio.

## Como aferir completude (derivada, não fixa)
Não há "número mágico" de linhas. Calcule o esperado a partir do que existe:
1. Descubra os conjuntos presentes: `M` = métodos, `N` = normalizações,
   `S` = segmentos, `K` = métricas (todos via `cut … | sort -u`).
2. A grade deveria ser o **produto cartesiano cheio**: cada `(method × norm)`
   aparece em todos os `S × K`. Esperado = `|M|·|N|·|S|·|K|` linhas de dados.
3. **Reporte o que falta** relativo a esse produto: pares `(method,norm)` ausentes,
   métricas que faltam para algum par, ou um segmento incompleto. Diferença entre o
   nº de linhas real e o produto = lacuna a listar (não a esconder).
4. Se métodos/normalizações **novos** aparecerem em relação à execução anterior,
   apenas inclua-os — é o comportamento esperado.

## O que produzir
### 1. Tabelas no formato do artigo (Tabelas 2–5 de França et al.)
Cada linha é um **par (norm × fusão)** *presente no CSV*; as colunas são as
métricas **segmentadas cabeça/cauda**. Replique a geometria do artigo:
- Uma tabela **por dataset** (e, se pedido, por valor de k).
- Colunas agrupadas por segmento — blocos `head` / `tail` (e `overall` se couber),
  com `\cmidrule` entre blocos. **Sempre inclua a cauda.**
- Linhas = **todos os pares presentes** (não uma sublista fixa). Se o Daniel pedir
  recorte (ex.: top-N por uma métrica), **declare o recorte** numa nota.
- Marque a linha do **baseline CombMNZ+ZMUV** (`method=combmnz, norm=zmuv`) **se
  presente** — é a referência do artigo. Marque-a, não a chame de "melhor".
- Ordene como o artigo (ou por uma métrica de cauda, ex.: `tail ndcg@5`), mas
  **declare o critério de ordenação** numa nota — não o esconda.
- **booktabs** (`\toprule/\midrule/\bottomrule`); números como `mean` (sem o `std`
  na célula, salvo se pedido — aí `mean{\scriptsize ±std}`). Não arredonde além das
  casas do artigo (3–4); diga quantas casas usou.

### 2. Relatório de completude (sempre, antes das tabelas)
- **Quais datasets têm `gridsearch.csv`** (e se veio de `data/` ou do backup).
- **Nº de linhas real vs. produto cartesiano esperado** (calculado acima); liste
  pares/métricas faltantes.
- **Quais folds** sustentam cada média, se rastreável (Eurlex/Wiki10 = 5 folds;
  datasets grandes = **3 dos 5**, folds 0,1,2). Se o CSV não registra o nº de folds,
  **diga isso** e não o invente.
- **O que está PENDENTE**: dataset sem CSV, denso ainda não inferido, PSP@k/PSnDCG@k
  ausente (se a família ainda não estiver no CSV). Cada pendência vira placeholder:
  `\todo{AmazonCat denso: pendente}` — nunca célula vazia silenciosa.

## Regras (alinhadas aos guardrails)
- **Determinístico:** os números saem do CSV, byte a byte. Se precisar reagregar,
  **mostre a conta** e prefira reusar o `mean/std` já calculado.
- **Sem juízo:** não escreva "outperforms", "best", "significativo". Só apresente.
- **Cauda obrigatória:** nenhuma tabela só de `overall`.
- **Honestidade de cobertura:** se cortou linhas (top-N), **declare** quantas e por
  quê. Nada de truncar silenciosamente.
- **Sem estatística inventada:** o CSV traz `mean,std` entre folds — **não**
  fabrique p-valor, IC ou tamanho de efeito.

## Formato de saída (exato)
1. **## Completude** — o relatório acima (datasets, linhas vs. produto esperado,
   folds, pendências, e métodos/normalizações novos detectados).
2. **`````latex`** — as tabelas `booktabs`/`acmart` prontas para colar, com a linha
   do baseline marcada (se presente), o critério de ordenação em nota e `\todo{}`
   nos buracos.
3. **## Procedência** — para cada número-chave, de qual CSV/linha veio (caminho +
   `method,norm,segment,metric`), para o `verificador-numerico` reconferir.
4. **## Pendente** — lista do que falta medir para a tabela ficar completa.

## Restrições (invioláveis)
- Não rode experimento, não chame recuperador/fusão; só lê CSV (e `wc`/`cut` para
  inspecionar).
- Não invente célula, fold, dataset, método ou métrica ausente — vira `\todo{}`.
- **Não crave a lista de fusões/normalizações** — derive-a do CSV (grade aberta).
- Não interprete nem ranqueie por mérito científico além de ordenar+marcar.
- A skill é em português; as tabelas/rótulos do paper, em inglês.
