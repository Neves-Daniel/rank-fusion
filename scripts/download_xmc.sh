#!/usr/bin/env bash
# Baixa um dataset do PECOS xmc-base (archive.org) e o converte para o layout que
# data.py espera. Genérico: serve para qualquer slug do xmc-base.
#
#   Uso:  bash scripts/download_xmc.sh <slug>
#   Ex.:  bash scripts/download_xmc.sh amazoncat-13k
#         bash scripts/download_xmc.sh wiki10-31k
#         bash scripts/download_xmc.sh amazon-670k
#
# O <slug> é tanto o nome no archive.org quanto a pasta local: data/<slug>/raw,
# então rode os CLIs com --dataset <slug> (ex.: --dataset amazoncat-13k).
#
# Layout do tarball (xmc-base/<slug>/): X.trn.txt, X.tst.txt (texto, 1 doc/linha),
# Y.trn.npz, Y.tst.npz (matriz docs×rótulos), output-items.txt (vocabulário). Os
# X.*.npz (TF-IDF) são ignorados. ATENÇÃO: AmazonCat/Amazon são grandes (vários GB
# baixados; ~1,5M docs no AmazonCat) — garanta espaço em /data e paciência no download.
set -euo pipefail

SLUG="${1:?uso: bash scripts/download_xmc.sh <slug>  (ex.: amazoncat-13k)}"
DEST="data/$SLUG/raw"
TARURL="https://archive.org/download/pecos-dataset/xmc-base/$SLUG.tar.gz"
TMP="data/$SLUG/_dl"

# baixa URL -> arquivo, usando o que existir (o container da Brev não tem curl)
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

if [ -s "$DEST/trn_X.txt" ] && [ -s "$DEST/Y.trn.npz" ] && [ -s "$DEST/Y.txt" ]; then
    echo ">> $SLUG já preparado em $DEST"
    ls -lh "$DEST"
    exit 0
fi

TARBALL="$TMP/$SLUG.tar.gz"
if [ ! -s "$TARBALL" ]; then
    echo ">> Baixando $SLUG.tar.gz do xmc-base (archive.org) ... (pode demorar)"
    fetch "$TARURL" "$TARBALL"
fi

echo ">> Extraindo ..."
tar -xzf "$TARBALL" -C "$TMP"

SRC="$TMP/xmc-base/$SLUG"
if [ ! -d "$SRC" ]; then
    echo "!! Pasta esperada não encontrada: $SRC" >&2
    echo "   Confira o slug e o conteúdo do tarball:" >&2
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
echo "Valide as estatísticas com: python -m src.data $DEST"
