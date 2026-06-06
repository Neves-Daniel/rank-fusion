---
description: Salva uma memória persistente na subpasta de categoria correta (padrão avançado)
allowed-tools: Read, Write, Edit, Bash(ls:*), Bash(find:*)
---

Salve uma memória persistente do projeto seguindo o **padrão avançado de pastas
por categoria**. A memória vem de: $ARGUMENTS (se vazio, infira o fato relevante
da conversa atual).

## Diretório de memória (caminho absoluto fixo)
`/home/dnpin/.claude/projects/-home-dnpin-nlp-rank-fusion/memory/`

Este é o diretório que o harness carrega a cada sessão — **não** use `.claude/memory/`.

## Passos
1. **Identifique a categoria** da memória, entre as pastas existentes:
   - `projeto/` — estado/objetivo/decisões do XMTC rank-fusion (type: project)
   - `usuario/` — quem é o usuário + feedback de como trabalhar (type: user | feedback)
   - `ambiente/` — execução, infraestrutura, Brev/Docker, limites de hardware (type: project | feedback | reference)
   - Se nenhuma servir, crie uma nova pasta com nome curto em kebab-case.
2. **Antes de criar, verifique duplicata:** leia o `README.md` da categoria e os
   arquivos relacionados. Se já existe memória que cobre o fato, **edite-a** em vez
   de criar outra.
3. **Crie a pasta** `{categoria}/` se não existir.
4. **Crie o arquivo** `{categoria}/{slug}.md` com frontmatter:
   ```markdown
   ---
   name: <slug-kebab-case>
   description: <resumo de uma linha — usado para recall de relevância>
   metadata:
     type: user | feedback | project | reference
   ---

   <o fato. Para feedback/project, siga com **Why:** e **How to apply:**.
   Ligue memórias relacionadas com [[name]] (o slug do frontmatter, não o caminho).>
   ```
   O `name:` e o nome do arquivo devem coincidir (para os `[[links]]` resolverem).
5. **Atualize o índice da categoria** `{categoria}/README.md` com uma linha:
   `- [Título](slug.md) · gancho curto`. Mantenha-o com menos de 30 linhas.
6. **Atualize `MEMORY.md` raiz** apenas se a categoria for **nova** (adicione a
   linha `- [Categoria](categoria/README.md) · ...`). Mantenha-o sob 50 linhas.

## Regras
- Um fato por arquivo. Não duplique o que o repositório já registra (código,
  histórico git, CLAUDE.md) — nesse caso, salve só o que foi **não-óbvio**.
- Converta datas relativas em absolutas.
- Nunca grave segredos/credenciais.
- Ao terminar, confirme ao usuário o caminho do arquivo salvo e em qual categoria.
