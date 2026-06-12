# Integração UDLF/pyUDLF — re-ranking e agregação contextual de rankings

**Origem:** e-mail do grupo do Prof. Daniel Pedronette (UNICAMP/UNESP, jun/2026)
oferecendo colaboração. O método FUR (Fusion Regression) deles é para fusão de
*regressores* (não se aplica), mas indicaram o framework **UDLF** (C++) e o wrapper
**pyUDLF** (Python), com 11 métodos de re-ranking e agregação de rankings
não-supervisionados. Métodos sugeridos no e-mail: **RFE, LHRR e CPRR**.
Ofereceram contato com doutorandos habituados ao framework.

- UDLF: <https://github.com/UDLF/UDLF> · <https://www.ic.unicamp.br/~dcarlos/UDLF/index.html>
- pyUDLF: <https://github.com/UDLF/pyUDLF>
- RFE: Valem, Pedronette & Latecki, "Rank Flow Embedding for Unsupervised and
  Semi-Supervised Manifold Learning", IEEE TIP 32: 2811-2826 (2023)
- LHRR: Pedronette, Valem, Almeida & Torres, "Multimedia Retrieval Through
  Unsupervised Hypergraph-Based Manifold Ranking", IEEE TIP 28(12): 5824-5838 (2019)
- CPRR: Valem, Pedronette & Almeida, "Unsupervised similarity learning through
  Cartesian product of ranking references", Pattern Recognit. Lett. 114: 41-52 (2018)

## Por que interessa ao projeto

Os métodos do UDLF são uma **família distinta** dos fundidores do ranx
(CombSUM/CombMNZ/RRF/...): em vez de combinar escores/postos par-a-par, eles
exploram a **estrutura contextual** das listas (grafos de kNN recíproco,
hipergrafos, embeddings de fluxo de ranking) para *aprender* uma nova similaridade
sem supervisão. Avaliá-los ao lado da grade ranx (mesmos folds, mesmas métricas
cabeça/cauda) amplia a contribuição do estudo: "fusão clássica × fusão contextual"
para tail labels em XMTC.

## Fatos verificados (código-fonte, jun/2026)

1. **Métodos (12 nomes no config, 11 reais + NONE):** CPRR, RLRECOM, RLSIM,
   CONTEXTRR, RECKNNGRAPH, RKGRAPH, CORGRAPH, LHRR, BFSTREE, RDPAC, RFE.
2. **Duas tarefas:** `UDL_TASK = UDL` (re-ranking de 1 conjunto de listas) e
   `UDL_TASK = FUSION` (agregação de N conjuntos: `NUM_INPUT_FUSION_FILES` +
   `INPUT_FILES_FUSION_1..N`, 1-indexado). Em FUSION (visto em `Cprr.cpp`), o
   método roda sobre **cada** arquivo, acumula as matrizes de similaridade
   aprendidas (`matrixAgg += matrix`) e re-ordena ao final.
3. **Formato de entrada (RK/NUM):** linha *i* = lista ranqueada do elemento *i*,
   top-L índices separados por espaço. Exige tb. `lists file` (1 identificador por
   linha; define n e o mapeamento índice→nome). Saída no mesmo formato (sem
   escores — só a ordem).
4. **Cenário "quadrado" (premissa central do framework):** as listas são do
   conjunto **sobre ele mesmo** (CBIR: cada imagem é query e item). A matriz de
   similaridade interna aloca **n² floats** (`Udl::initSparseMatrix` faz
   `new float[n*n]`) → n por execução limitado a ~10–40K elementos (0,4–6,4 GB).
5. **pyUDLF:** wrapper leve (deps: `requests`, `numpy`); baixa binário Linux
   pré-compilado para `~/.pyudlf/bin/` no primeiro uso (`http://udlf_linux.lucasvalem.com`).
   Alternativa reprodutível: compilar o UDLF do fonte (tem `Makefile`) no
   `Dockerfile` da imagem `rank-fusion:latest` e apontar `setBinaryPath`.
   API: `InputType()` + `set_param(...)` + `run_calls.run(input, get_output=True)`;
   saída ranqueada em `output.rk_path`.

## O descompasso e a adaptação proposta

Nosso cenário é **bipartido** (queries = documentos, itens = rótulos); o UDLF supõe
listas item→item. O e-mail do Pedronette já avisa: a agregação contextual vale
"considerando que há rankings disponíveis **para os itens** nos rankings" — ou
seja, precisamos de **rankings rótulo→rótulo** além dos doc→rótulo que já temos.

**Adaptação por blocos por query** (a validar com os doutorandos do grupo):

1. Para cada query *q* do fold: bloco B(q) = {q} ∪ candidatos (união dos top-128 do
   esparso e do denso; ≤ 257 elementos).
2. Listas do bloco: a linha de *q* é o ranking doc→rótulo (do recuperador ou da
   fusão a re-ranquear); a linha de cada rótulo é o ranking rótulo→rótulo
   **restrito ao bloco** (fonte: ver decisão abaixo).
3. Empacotar **vários blocos disjuntos numa única execução** (block-diagonal):
   como os métodos só usam o top-L/kNN das listas, blocos que não se referenciam
   não interagem (verificar com `scripts/smoke_udlf.py`). Lote de ~50–100 queries
   → n ≈ 13–26K por chamada (n² cabe na RAM), chamadas paralelas nos 255 cores.
4. Extrair as linhas das queries da saída → TREC (escore sintético `1/(rank+1)`,
   as métricas só usam a ordem) → `metrics.py`/`gridsearch.py` como qualquer run.

### Decisão em aberto: fonte do ranking rótulo→rótulo

| Opção | Fonte | Custo | Observação |
|---|---|---|---|
| A | Co-ocorrência em Y_train do fold (cosseno entre colunas) | offline, sem GPU | fold-safe; clássico em XMC; **recomendada p/ 1ª passada** |
| B | BM25 sobre nomes/descrições RAG-labels | offline, leve | descrições já existem por fold (Eurlex) |
| C | Embeddings do encoder denso | exige re-rodar denso | pesos não são salvos hoje; mudar `retrieve_dense.py` p/ exportar embeddings de rótulos |

### Modos de avaliação (ambos comparáveis à grade ranx — mesmos folds/candidatos)

1. **FUSION(esparso, denso)** com cada método UDLF → linhas novas na tabela de
   fusão, lado a lado com (norm × fusão) do ranx.
2. **UDL re-ranking** sobre o melhor run fundido do ranx (ex.: CombMNZ+ZMUV) →
   "fusão clássica + re-ranking contextual".

## Plano incremental

1. `pip install git+https://github.com/UDLF/pyUDLF.git` (local) +
   `python scripts/smoke_udlf.py` — valida binário, formatos, FUSION e a premissa
   de blocos independentes. **(pronto para rodar)**
2. Implementar `src/udlf_fusion.py`: TREC runs → blocos → arquivos RK/NUM →
   pyUDLF → TREC out (com `--dataset`, `--folds`, fonte rótulo→rótulo plugável).
3. Eurlex-4K, fold 0, CPRR/LHRR/RFE nos dois modos; avaliar com `metrics.py`
   (cabeça/cauda) e comparar com a grade ranx.
4. Se promissor: 5 folds Eurlex → Wiki10 → grandes (protocolo 3-dos-5 folds).
5. Responder o e-mail: perguntar aos doutorandos como eles adaptam o framework a
   cenários bipartidos/textuais (pode existir receita pronta do grupo).

## Honestidade científica

- Os métodos foram desenhados para CBIR (cenário quadrado); a adaptação bipartida
  por blocos + rankings rótulo→rótulo derivados é **nossa** e deve ser descrita
  explicitamente (e idealmente validada com o grupo da UNICAMP).
- A saída do UDLF não tem escores → só a ordem é comparável; registrar.
- Parâmetros dos métodos (K, L, T) têm defaults de CBIR (K=20, L=400–1400);
  precisam de ajuste para blocos de ~257 elementos (ex.: L = tamanho do bloco).
