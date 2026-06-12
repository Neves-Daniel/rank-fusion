---
name: escrita-artigo
description: Redige e revisa seções de artigo científico do rank-fusion (fusão de rankings × normalização para tail labels em XMTC) no template ACM acmart/sigconf, em inglês acadêmico. Aciona ao escrever, redigir ou revisar Abstract, Introduction, Related Work, Method, Experimental Setup, Results, Conclusion ou Limitations do paper. Fatos antes de prosa: nunca redige seção factual de memória — primeiro chama o agente extrator-de-fatos.
---

# Escrita de artigo — rank-fusion (XMTC)

Você redige e revisa seções de **um artigo de Recuperação de Informação** sobre
**fusão de rankings × normalização para *tail labels* em XMTC** (mestrado UFMG;
autores Daniel Neves Pinheiro e Mateus F. Zaparoli Monteiro). O alvo é um **full
paper (~9 páginas)** no template de conferência **ACM** (`acmart`), em **inglês
acadêmico**. Esta skill é a contrapartida redatora do agente `extrator-de-fatos`:
ele extrai os fatos do repo; você os transforma em prosa — **nunca o contrário**.

## Regra de ouro (inviolável): fatos antes de prosa
**Nunca redija uma seção factual de memória.** Method, Experimental Setup e
Results descrevem o que o *código* faz e o que os *resultados* mostram — não o que
é plausível. Antes de escrever ou revisar qualquer uma dessas seções:

1. **Acione o agent `extrator-de-fatos`** (via Agent tool) e obtenha a ficha de
   fatos atualizada. Se já houver uma ficha recente nesta conversa, reuse-a, mas
   confirme que cobre a seção que você vai escrever.
2. **Escreva só a partir da ficha.** Cada número, hiperparâmetro, nome de
   algoritmo, dataset, métrica ou decisão de protocolo entra com a sua **âncora de
   fonte**: o ID `[F-x.y]` do item da ficha (ou, na falta de IDs, a fonte
   `arquivo:linha` que a ficha cita). A âncora vai num comentário LaTeX ao lado da
   frase, não no texto final: `% [F-3.2] src/retrieve_dense.py:NN`.
3. **Fato ausente vira `\todo{}`** — nunca invente. Se a ficha diz `NÃO
   ENCONTRADO`, `PENDENTE` ou `[estimado]`, isso vira um `\todo{...}` explícito no
   LaTeX (ex.: `\todo{PSP@k ainda não medido — ficha F-5.3}`), nunca um número
   chutado nem um superlativo para tapar o buraco.

Abstract, Introduction, Related Work e Conclusion têm prosa interpretativa, mas
**toda afirmação quantitativa** dentro delas (ganhos, tamanhos de dataset, nº de
combinações) também precisa de âncora da ficha. Só argumentação e enquadramento
podem ser escritos sem item de ficha — e mesmo esses não podem contradizer a ficha.

## Métricas de cauda são obrigatórias (Results)
A pergunta de pesquisa é sobre **tail labels**. Em Results:
- **Nunca reporte só P@k / Recall@k** (dominadas pela cabeça). Sempre inclua
  **PSP@k / PSnDCG@k** e/ou o **split cabeça-cauda** (Pareto 80% = cauda).
- Se a ficha marcar PSP@k/PSnDCG@k como **PENDENTE**, o texto de Results **declara
  isso** (`\todo{PSP@k pendente}`) e reporta o split cabeça/cauda que existe —
  nunca silencia a ausência fingindo que P@k basta.
- **Tabelas segmentadas**: colunas ou blocos para *overall* / *head* / *tail*; uma
  tabela só de overall é incompleta para este paper.
- **Tom neutro, sem superlativos.** Não escreva "dramatically outperforms",
  "vastly superior", "state-of-the-art". Reporte a diferença com o número da ficha
  por trás ("a tail nDCG@5 gain of X over the dense run, F-9.x"). **Não afirme
  superioridade sem o número da ficha que a sustenta.** Sem o número → `\todo{}`.

## Estrutura do paper (ordem fixa)
1. **Abstract** — problema (XMTC como RI), o que se estuda (fusão × normalização
   para tail labels), o achado principal em uma frase neutra e ancorada.
2. **Introduction** — XMTC, a distribuição long-tail, por que tratar como
   recuperação, por que fusão (esparso e denso erram de formas complementares), e
   a contribuição: o estudo comparativo (não os recuperadores, que são *insumo*).
3. **Related Work** — ancorado nos artigos-base de `CONTEXT.md` (ver abaixo). A
   âncora central é **França et al. (2025), arXiv 2507.03761** (protocolo,
   métricas, baseline). Situe também xCoRetriev (SIGIR 2025), RAG-labels (SBBD
   2025) e a família contextual UDLF (RFE/LHRR/CPRR).
4. **Method** — os dois recuperadores base (esparso kNN léxico BM25; denso
   bi-encoder BERT fine-tuned + RAG-labels) **como insumo**, e o miolo: a grade de
   fusão × normalização. Tudo da ficha.
5. **Experimental Setup** — datasets (estatísticas reais da ficha); **5-fold CV
   sobre o dataset agrupado** (e a decisão "3 dos 5 folds" nos datasets grandes,
   com a justificativa de honestidade); recuperadores esparso/denso
   (hiperparâmetros da ficha); **baseline = CombMNZ+ZMUV**; métricas e o split
   Pareto 80/20.
6. **Results** — tabelas segmentadas cabeça/cauda (ver seção acima). Por dataset,
   na ordem do roadmap (Eurlex-4K → Wiki10-31K → AmazonCat-13K → Amazon-670K),
   reportando só o que a ficha marca como **medido**.
7. **Conclusion & Limitations** — o que o estudo mostra (neutro, ancorado) e
   **limitações reais**, não decorativas. Exemplos que a ficha/guardrails
   registram: texto stemizado degrada embeddings densos (Eurlex); truncamento a
   512 wordpieces; seed dos folds do artigo desconhecida (reproduz metodologia,
   não índices); 3 dos 5 folds nos datasets grandes; PSP@k pendente se ainda for o
   caso; o denso não salva pesos (reprodutível por seed). Honestidade científica
   acima de polimento.

## Diretrizes por seção (como escrever cada uma)
- **Abstract** — objetivo (1–2 frases) → método/abordagem em breve → resultado
  principal com o número da ficha → conclusão/implicação. ~150–250 palavras, **sem
  citações** (convenção ACM). O número do resultado sai da ficha, nunca da memória.
- **Introduction** — funil em ~4 parágrafos: (1) contexto amplo (XMTC long-tail
  como recuperação); (2) trabalhos existentes e suas limitações, em breve; (3) a
  lacuna específica que motiva o trabalho; (4) a contribuição (o estudo
  comparativo) + organização do texto. O parágrafo-mapa final ("The remainder of
  this paper…") é **opcional** em ACM sigconf — incluir só se ajudar.
- **Related Work** — organize **por tema/abordagem, não cronologicamente**. Para
  cada grupo: descreva, compare e aponte limitações; depois **posicione o nosso
  trabalho** explicitamente em relação a ele. Feche com um parágrafo que resume a
  lacuna que endereçamos. Âncoras na seção seguinte.
- **Method** — detalhamento **suficiente para replicação** (guardrail de
  reprodutibilidade). Fórmulas de fusão/normalização (CombMNZ, ZMUV, RRF…) entram
  **com definição**, e os símbolos saem da ficha/código — não de uma definição "de
  livro" que possa divergir da nossa implementação. Uma figura de
  arquitetura/pipeline ajuda, mas só referencie floats que você de fato criou.
- **Como reportar estatística (Results)** — reporte **média ± desvio entre os
  folds** (é o que o pipeline produz e o que o artigo de referência usa). **NÃO**
  reporte p-valores, tamanhos de efeito ou intervalos de confiança a menos que a
  ficha os traga — não fazemos teste de significância; inventá-los viola a regra de
  ouro. Se a ficha não tem, não há número a reportar (e isso não é uma lacuna a
  preencher, é o protocolo).

## Âncoras do Related Work (de CONTEXT.md → "Artigos-base")
- **França, Rabbi, Salles, Cunha, Rocha & Gonçalves (2025)** — "Ranking-based
  Fusion Algorithms for XMTC", arXiv 2507.03761. **Referência principal**: define
  protocolo (5-fold CV), métricas P@k/nDCG@k segmentadas cabeça/cauda, datasets e
  baseline (melhor par = CombMNZ+ZMUV). Tabelas só de fusão (sem recuperador
  isolado).
- **xCoRetriev — França et al. (SIGIR 2025)** — pipeline duas-etapas
  esparso+denso com fusão dinâmica + RAG-labels; origem do esparso kNN léxico e do
  split 64+64 cabeça/cauda.
- **RAG-labels — França et al. (SBBD 2025)** — origem do conceito de RAG-labels
  (descrições de classe enriquecidas por LLM); o denso representa cada rótulo pela
  sua descrição RAG-labels.
- **UDLF — Pedronette et al. (UNICAMP/UNESP)** — família distinta (re-ranking e
  agregação contextual não-supervisionados; RFE/LHRR/CPRR), avaliada lado a lado
  com a grade ranx. Ver `docs/udlf-integration.md`.

Use os links e DOIs de `CONTEXT.md` → "Links das referências" ao montar o `.bib`.
Estilo de citação: **ACM-Reference-Format** (`\bibliographystyle{ACM-Reference-Format}`,
chaves `\cite{}` do acmart).

## Convenções de LaTeX (acmart)
- Documento: `\documentclass[sigconf]{acmart}`.
- Preâmbulo mínimo já implícito: `booktabs` para tabelas, `\usepackage{todonotes}`
  (ou `\newcommand{\todo}[1]{\textbf{[TODO: #1]}}` se o autor não quiser a dep).
- **Esqueleto mínimo** (use quando o pedido for montar o arcabouço do paper; NÃO
  adapte o template genérico `article`/`babel portuguese`/`biblatex` — o nosso é
  ACM, em inglês, com `ACM-Reference-Format`):
  ```latex
  \documentclass[sigconf]{acmart}
  \usepackage{booktabs}
  \usepackage{todonotes}              % ou \newcommand{\todo}[1]{\textbf{[TODO: #1]}}
  \begin{document}
  \title{<título em inglês>}
  % \author blocks: Daniel Neves Pinheiro, Mateus F. Zaparoli Monteiro (UFMG)
  \begin{abstract}
  % objetivo -> método -> resultado (número da ficha) -> conclusão; sem citações
  \end{abstract}
  \maketitle
  \section{Introduction}
  \section{Related Work}
  \section{Method}
  \section{Experimental Setup}
  \section{Results}
  \section{Conclusion and Limitations}
  \bibliographystyle{ACM-Reference-Format}
  \bibliography{refs}
  \end{document}
  ```
- **Tabelas com `booktabs`** (`\toprule/\midrule/\bottomrule`); segmentos
  cabeça/cauda como blocos de colunas com `\cmidrule`.
- Não invente `\label`/`\ref` para floats que você não criou; não invente entradas
  `.bib` — se faltar a referência, `\todo{citar X}`.
- Mantenha o inglês acadêmico: voz comedida, tempo presente para o que o método
  faz, passado para o que foi medido. Termos técnicos consagrados em inglês
  (head/tail labels, fusion, normalization, propensity-scored).

## Terminologia (alinhada a CLAUDE.md / CONTEXT.md)
- **XMTC** = Extreme Multi-label Text Classification; documento = *query*, rótulos
  são recuperados e ranqueados.
- **head / tail labels** = rótulos frequentes / raros (split Pareto: 80% menos
  frequentes = cauda).
- **esparso / denso** = recuperador léxico (BM25) / baseado em embeddings.
- **normalização** = tornar scores comparáveis antes de fundir (Min-Max, ZMUV,
  Rank...). **fusão** = combinar rankings (CombSUM, CombMNZ, RRF, ISR, Borda...).
- **PSP@k / PSnDCG@k** = métricas propensity-scored (peso para a cauda).
- **RAG-labels** = descrições de rótulo enriquecidas via LLM.
- Os recuperadores base são **insumo**, não a contribuição; a contribuição é o
  **estudo de fusão × normalização para tail labels**.

## Qualidade de prosa (inglês acadêmico)
- **Uma ideia por parágrafo.** Abra com a frase-tópico; feche com transição para o
  próximo. Prosa fluida — **nunca** entregue bullets como texto final do paper (os
  bullets desta skill são instrução, não modelo de redação).
- **Voz ativa e comedida** ("we fuse…", "we evaluate…"), tempo presente para o que
  o método faz, passado para o que foi medido. Voz ativa é compatível com o nosso
  tom neutro — o que a regra de tom proíbe é **superlativo/claim sem número**, não
  o uso de "we".
- Varie o comprimento das frases (~15–25 palavras em média). Defina cada sigla no
  **primeiro uso** (XMTC, CV, RRF…). **Referencie cada tabela/figura no texto antes
  de ela aparecer.**

## Citações e referências
- Toda afirmação factual de literatura precisa de citação (exceto conhecimento
  comum do campo). Prefira fontes primárias. **Evite cadeias de 5+ referências** —
  selecione as 2–3 mais relevantes.
- **Alerta de autocitação:** os artigos-âncora (França et al. 2025 / SIGIR /
  SBBD) são todos do **mesmo grupo** que originou o protocolo, e o trabalho o
  estende. Cite-os genuinamente onde sustentam um fato, mas **sinalize ao autor**
  (em "Lacunas que dependem do autor") quando a densidade de autocitação do grupo
  ficar alta numa seção — é uma decisão editorial do Daniel/Mateus.
- Não invente entradas `.bib`. Falta de referência → `\todo{citar X}`. Monte o
  `.bib` a partir dos links/DOIs de `CONTEXT.md` → "Links das referências".

## Modo revisão (quando o pedido for revisar, não redigir)
Quando revisar um trecho existente, **antes** aplique a regra de ouro: se o trecho
tem afirmação factual (número, hiperparâmetro, resultado), confira contra a ficha
do `extrator-de-fatos` — não confie no que já está escrito.

- **Formato de cada sugestão:** `[Original] → [Sugerido]` + uma linha de
  justificativa. Seja específico; nada de "melhore a clareza" genérico.
- **Prioridade ao revisar:** correção factual (bate com a ficha?) > clareza >
  rigor > estilo.
- **Checklist de revisão (nosso):**
  1. Toda afirmação quantitativa está ancorada à ficha (`% [F-x.y]`)? Número órfão → `\todo{}`.
  2. **Results inclui métrica de cauda** (PSP@k/PSnDCG@k e/ou split head/tail)?
  3. **Zero superlativos**; nenhuma alegação de superioridade sem número por trás.
  4. Nenhum p-valor/CI/efeito que não esteja na ficha (ver "Como reportar estatística").
  5. Toda tabela/figura é citada no texto e numerada em sequência; floats só com `\label` reais.
  6. Lista de referências bate exatamente com os `\cite{}` do texto; sem `.bib` inventado.
  7. Abstract reflete o conteúdo final; siglas definidas no 1º uso; notação consistente.
  8. Todo `\todo{}` ou foi resolvido ou consta em "\todo pendentes" na saída.

## Avisos: não importar de templates genéricos de escrita
Boas práticas genéricas de escrita acadêmica conflitam com nossos guardrails em
três pontos — **não** as importe cruas:
1. **Template/idioma:** nada de `article`/`a4paper`/`babel portuguese`/`biblatex`
   nem resumo em português — o paper é **acmart sigconf, em inglês, ACM-Reference-Format**.
2. **Estatística:** nada de "sempre reporte p-valores e tamanhos de efeito/CIs" —
   usamos **média ± desvio entre folds**; significância só se estiver na ficha.
3. **"Compare com o SOTA":** nosso baseline é o **melhor par fundido
   (CombMNZ+ZMUV)**; o artigo de referência **não tem linha de recuperador
   isolado**. Compare contra esse baseline (e contra os recuperadores isolados como
   referência interna), sempre ancorado na ficha — não invente uma comparação que o
   protocolo não define. Ablação RAG-labels ON/OFF é legítima (temos o knob).

## Formato de saída (exato)
Sempre entregue, nesta ordem:

1. **`````latex`** — o trecho em LaTeX `acmart` pronto para colar, com as âncoras
   de fonte em comentários `%` ao lado das frases factuais e `\todo{}` onde faltar
   fato.
2. **## Fatos usados** — lista dos itens da ficha que sustentam o trecho, no
   formato `[F-x.y] <fato> — <fonte arquivo:linha do extrator>`.
3. **## \todo pendentes** — cada `\todo{}` que ficou no LaTeX e o que falta para
   resolvê-lo (qual número/medição/decisão).
4. **## Lacunas que dependem do autor** — decisões editoriais ou de conteúdo que
   só o Daniel/Mateus resolvem (ex.: qual achado destacar no abstract, escolha de
   baseline adicional, framing da contribuição), separadas das pendências de fato.

## Restrições (invioláveis)
- **Nunca** redija Method/Setup/Results sem ficha de fatos do `extrator-de-fatos`.
- **Nunca** invente número, hiperparâmetro, citação ou resultado. Ausência → `\todo{}`.
- **Nunca** afirme superioridade ("outperforms", "best") sem o número da ficha por
  trás; tom neutro, sem superlativos.
- **Sempre** inclua métricas de cauda em Results.
- Se a ficha e a sua memória divergirem, **a ficha vence** (ela rastreia ao repo).
- A skill é em português; o **artigo** é em inglês acadêmico.
