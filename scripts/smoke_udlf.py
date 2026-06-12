"""Smoke test do UDLF/pyUDLF com dados sintéticos minúsculos (CPU, ~segundos).

Valida, antes de qualquer integração real (ver docs/udlf-integration.md):
  1. pyUDLF instala/baixa o binário e executa (UDL_TASK=UDL, método CPRR);
  2. agregação de 2 rankings funciona (UDL_TASK=FUSION);
  3. PREMISSA DE BLOCOS INDEPENDENTES: blocos disjuntos empacotados numa única
     execução (block-diagonal) não vazam elementos entre si — é o que permite
     processar lotes de queries por chamada na adaptação bipartida do XMTC.

Uso (após instalar):  pip install "git+https://github.com/UDLF/pyUDLF.git"
                      python scripts/smoke_udlf.py
"""

import sys
import tempfile
from pathlib import Path

from pyUDLF import run_calls as udlf
from pyUDLF.utils import inputType

# Dois blocos disjuntos de 10 elementos: A = 0..9, B = 10..19.
# Listas ranqueadas só referenciam o próprio bloco (L = tamanho do bloco).
BLOCK = 10
N = 2 * BLOCK
L = BLOCK


def block_of(i: int) -> range:
    start = (i // BLOCK) * BLOCK
    return range(start, start + BLOCK)


def make_rk_file(path: Path, rotate: int) -> None:
    """Linha i = [i] + demais elementos do bloco rotacionados (determinístico).

    `rotate` muda a ordem dos vizinhos → simula rankers distintos p/ o FUSION.
    """
    lines = []
    for i in range(N):
        others = [j for j in block_of(i) if j != i]
        others = others[rotate:] + others[:rotate]
        lines.append(" ".join(str(x) for x in [i] + others))
    path.write_text("\n".join(lines) + "\n")


def read_rk_file(path: Path) -> list[list[int]]:
    rows = []
    for line in path.read_text().strip().splitlines():
        rows.append([int(x) for x in line.split()])
    return rows


def check_blocks(rows: list[list[int]], label: str) -> bool:
    """Top-L de cada linha deve estar contido no bloco do elemento da linha."""
    assert len(rows) == N, f"{label}: esperava {N} linhas, veio {len(rows)}"
    ok = True
    for i, row in enumerate(rows):
        leaked = set(row[:L]) - set(block_of(i))
        if leaked:
            print(f"  FAIL [{label}] linha {i}: vazou p/ outro bloco: {sorted(leaked)}")
            ok = False
    if ok:
        print(f"  PASS [{label}] blocos independentes (nenhum vazamento no top-{L})")
    return ok


def base_input(workdir: Path, out_name: str) -> inputType.InputType:
    inp = inputType.InputType()  # baixa binário+config p/ ~/.pyudlf na 1ª vez
    inp.set_method_name("CPRR")
    inp.set_dataset_size(N)
    inp.set_lists_file(str(workdir / "lists.txt"))
    inp.set_param("INPUT_FILE_FORMAT", "RK")
    inp.set_param("INPUT_RK_FORMAT", "NUM")
    inp.set_param("OUTPUT_FILE", "TRUE")
    inp.set_param("OUTPUT_FILE_FORMAT", "RK")
    inp.set_param("OUTPUT_RK_FORMAT", "NUM")
    inp.set_param("OUTPUT_FILE_PATH", str(workdir / out_name))
    inp.set_param("OUTPUT_LOG_FILE_PATH", str(workdir / f"{out_name}_log.txt"))
    inp.set_param("EFFECTIVENESS_EVAL", "FALSE")  # sem classes file
    inp.set_param("EFFICIENCY_EVAL", "FALSE")
    # Parâmetros de CBIR (K=20, L=400) não cabem em blocos de 10:
    inp.set_param("PARAM_CPRR_L", L)
    inp.set_param("PARAM_CPRR_K", 3)
    inp.set_param("PARAM_CPRR_T", 2)
    return inp


def find_output(workdir: Path, out_name: str) -> Path:
    for cand in (workdir / out_name, workdir / f"{out_name}.txt"):
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        f"saída '{out_name}' não encontrada em {workdir}: {list(workdir.iterdir())}"
    )


def main() -> int:
    workdir = Path(tempfile.mkdtemp(prefix="udlf_smoke_"))
    print(f"Diretório de trabalho: {workdir}")

    (workdir / "lists.txt").write_text("\n".join(str(i) for i in range(N)) + "\n")
    make_rk_file(workdir / "rk_a.txt", rotate=0)
    make_rk_file(workdir / "rk_b.txt", rotate=3)

    ok = True

    print("\n[1/2] UDL re-ranking (CPRR, 1 conjunto de listas)...")
    inp = base_input(workdir, "out_udl")
    inp.set_param("UDL_TASK", "UDL")
    inp.set_input_files(str(workdir / "rk_a.txt"))
    res = udlf.run(inp, get_output=True)
    if res is False:
        print("  FAIL: execução UDL falhou (ver logs acima)")
        ok = False
    else:
        ok &= check_blocks(read_rk_file(find_output(workdir, "out_udl")), "UDL")

    print("\n[2/2] FUSION (CPRR, 2 conjuntos de listas)...")
    inp = base_input(workdir, "out_fusion")
    inp.set_param("UDL_TASK", "FUSION")
    inp.set_input_files([str(workdir / "rk_a.txt"), str(workdir / "rk_b.txt")])
    res = udlf.run(inp, get_output=True)
    if res is False:
        print("  FAIL: execução FUSION falhou (ver logs acima)")
        ok = False
    else:
        ok &= check_blocks(read_rk_file(find_output(workdir, "out_fusion")), "FUSION")

    print(f"\n{'SMOKE TEST OK' if ok else 'SMOKE TEST FALHOU'} — artefatos em {workdir}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
