#!/usr/bin/env bash
# Baixa o Wiki10-31K (benchmark XMTC) do PECOS xmc-base (archive.org) e o converte
# para o MESMO layout que data.py espera (igual ao do Eurlex-4K).
#
# Diferença vs. Eurlex: NÃO há espelho drop-in (tipo thekop79). A fonte que traz
# texto cru + matrizes de rótulo + vocabulário num só tarball é o xmc-base do PECOS.
# O texto do Wiki10 são artigos da Wikipédia em PALAVRAS INTEIRAS (não stemizado) e
# bem mais longos que o Eurlex — bom para o denso, mas atenção ao truncamento (512
# wordpieces no denso) e ao risco de query longa no esparso (dedup_query_terms protege).
#
# Layout do tarball (xmc-base/wiki10-31k/): X.trn.txt, X.tst.txt (texto, 1 doc/linha),
# Y.trn.npz, Y.tst.npz (matriz docs×rótulos), output-items.txt (vocabulário de rótulos).
# Os X.*.npz (TF-IDF) são ignorados — não usamos features pré-computadas.
set -euo pipefail

DEST="data/wiki10-31k/raw"
TARURL="https://archive.org/download/pecos-dataset/xmc-base/wiki10-31k.tar.gz"
TMP="data/wiki10-31k/_dl"

# baixa URL -> arquivo de saída, usando o que estiver disponível (o container da Brev
# não tem curl; wget ou python sempre existem)
fetch() {
    local url="$1" out="$2"
    if command -v curl >/dev/null 2>&1; then
        curl -fSL --retry 3 -o "$out" "$url"
    elif command -v wget >/dev/null 2>&1; then
        wget -O "$out" "$url"
    else
        python - "$url" "$out" <<'PY'
import sys, urllib.request
urllib.request.urlretrieve(sys.argv[1], sys.argv[2])
PY
    fi
}

mkdir -p "$DEST" "$TMP"

# já convertido? então nada a fazer
if [ -s "$DEST/trn_X.txt" ] && [ -s "$DEST/Y.trn.npz" ] && [ -s "$DEST/Y.txt" ]; then
    echo ">> Wiki10-31K já preparado em $DEST"
    ls -lh "$DEST"
    exit 0
fi

TARBALL="$TMP/wiki10-31k.tar.gz"
if [ ! -s "$TARBALL" ]; then
    echo ">> Baixando wiki10-31k.tar.gz do xmc-base (archive.org) ..."
    fetch "$TARURL" "$TARBALL"
fi

echo ">> Extraindo ..."
tar -xzf "$TARBALL" -C "$TMP"

SRC="$TMP/xmc-base/wiki10-31k"
if [ ! -d "$SRC" ]; then
    echo "!! Pasta esperada não encontrada: $SRC" >&2
    echo "   Confira o conteúdo do tarball:" >&2
    tar -tzf "$TARBALL" | head -20 >&2
    exit 1
fi

echo ">> Convertendo para o layout do data.py em $DEST ..."
cp -f "$SRC/X.trn.txt"        "$DEST/trn_X.txt"
cp -f "$SRC/X.tst.txt"        "$DEST/tst_X.txt"
cp -f "$SRC/Y.trn.npz"        "$DEST/Y.trn.npz"
cp -f "$SRC/Y.tst.npz"        "$DEST/Y.tst.npz"
cp -f "$SRC/output-items.txt" "$DEST/Y.txt"

echo ">> Limpando temporários ..."
rm -rf "$TMP"

echo ">> Arquivos:"
ls -lh "$DEST"
echo
echo "Esperado (Tabela 1 do artigo / xmc-base): ~14.146 treino / ~6.616 teste / 30.938 rótulos."
echo "Valide as estatísticas com: python -m src.data $DEST"
