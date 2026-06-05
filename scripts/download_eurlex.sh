#!/usr/bin/env bash
# Baixa o Eurlex-4K (formato PECOS, com texto pré-processado) do espelho HuggingFace
# thekop79/EURLex-4K. Não precisa de autenticação nem de gdown.
set -euo pipefail

DEST="data/eurlex4k/raw"
BASE="https://huggingface.co/datasets/thekop79/EURLex-4K/resolve/main"

mkdir -p "$DEST"
echo ">> Baixando Eurlex-4K para $DEST ..."
for f in raw/trn_X.txt raw/tst_X.txt raw/Y.txt Y.trn.npz Y.tst.npz; do
    out="$DEST/$(basename "$f")"
    if [ -s "$out" ]; then
        echo "   (já existe) $out"
    else
        echo "   baixando $f ..."
        curl -fSL --retry 3 -o "$out" "$BASE/$f"
    fi
done

echo ">> Arquivos:"
ls -lh "$DEST"
echo
echo "trn_X.txt/tst_X.txt = textos (1 doc/linha) | Y.*.npz = matriz docs×rótulos | Y.txt = vocabulário de rótulos"
