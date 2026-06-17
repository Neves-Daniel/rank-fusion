"""Consolida os gridsearch.csv em relatório de completude + tabelas LaTeX.
Determinístico: tudo derivado do CSV (grade aberta). Não interpreta, só formata."""
import csv, os
from collections import defaultdict

DATASETS = [
    ("Eurlex-4K", "data/eurlex4k/results/gridsearch.csv", "5-fold CV, full fold"),
    ("Wiki10-31K", "data/wiki10-31k/results/gridsearch.csv", "5-fold CV, full fold"),
    ("AmazonCat-13K", "data/amazoncat-13k/results/gridsearch.csv", "20K-query sample, folds 0/1/2"),
]
RANK_SEG, RANK_MET = "tail", "ndcg@5"
NORM_ORDER = ["minmax", "minmaxinv", "max", "sum", "zmuv", "rank", "borda"]
BASE = ("combmnz", "zmuv")

def ckpt_of(path):
    """Caminho do checkpoint long-format (.ckpt.csv) correspondente ao CSV final."""
    base,ext=os.path.splitext(path); return f"{base}.ckpt{ext or '.csv'}"

def load(path):
    """Lê o CSV final (com header method,norm,segment,metric,mean,std) OU o checkpoint
    long-format (MESMAS 6 colunas, na mesma ordem, SEM header). Fareja pela 1ª célula."""
    cells = defaultdict(dict)  # (method,norm) -> {(seg,met):(mean,std)}
    M,N,S,K = set(),set(),set(),set()
    n=0
    with open(path) as fh:
        has_header = fh.readline().split(",",1)[0].strip()=="method"; fh.seek(0)
        if has_header:
            rows=((r["method"],r["norm"],r["segment"],r["metric"],r["mean"],r["std"])
                  for r in csv.DictReader(fh))
        else:  # checkpoint posicional: method,norm,seg,metric,mean,std
            rows=(r for r in csv.reader(fh) if len(r)==6)
        for m,nm,seg,met,mean,std in rows:
            cells[(m,nm)][(seg,met)]=(float(mean),float(std))
            M.add(m); N.add(nm); S.add(seg); K.add(met); n+=1
    return cells,M,N,S,K,n

def norms_sorted(N):
    pref=[x for x in NORM_ORDER if x in N]; return pref+sorted(N-set(pref))

for name,path,foldnote in DATASETS:
    print("="*70); print(f"# {name}  ({path})")
    # Escolhe a fonte mais recente: o checkpoint só sobrevive enquanto um run está em
    # andamento (é removido ao gravar o CSV final). Se o checkpoint existe E é mais novo
    # que o CSV final, ele é o run atual → preferi-lo (evita um CSV antigo "sombrear" um
    # checkpoint novo, ex.: rodada com --psp por cima de uma sem PSP).
    ck=ckpt_of(path)
    ck_ok=os.path.exists(ck); fin_ok=os.path.exists(path)
    if ck_ok and (not fin_ok or os.path.getmtime(ck)>os.path.getmtime(path)):
        if fin_ok:
            print(f"  checkpoint MAIS NOVO que o CSV final — usando o checkpoint (run em andamento): {ck}")
        else:
            print(f"  CSV final ausente — usando checkpoint: {ck}")
        path=ck; foldnote+="; partial (from checkpoint)"
    elif not fin_ok:
        print("  AUSENTE"); continue
    else:
        print(f"  usando CSV final: {path}")
    cells,M,N,S,K,n = load(path)
    bpath=os.path.join(os.path.dirname(path),"baselines.csv")     # sparse/dense isolados (eval_significance)
    base_cells=load(bpath)[0] if os.path.exists(bpath) else {}
    spath=os.path.join(os.path.dirname(path),"significance.csv")  # testes pareados (eval_significance)
    exp=len(M)*len(N)*len(S)*len(K)
    print(f"  métodos |M|={len(M)} | normalizações |N|={len(N)} | segmentos={sorted(S)} | métricas |K|={len(K)}")
    print(f"  pares (method×norm)={len(cells)} | linhas={n} | esperado |M|·|N|·|S|·|K|={exp} | {'OK' if n==exp else 'FALTAM '+str(exp-n)}")
    print(f"  métricas presentes: {sorted(K)}")
    print(f"  fold note: {foldnote}")
    # ranking por tail ndcg@5
    rk=sorted(cells.items(), key=lambda kv: -kv[1].get((RANK_SEG,RANK_MET),(float('-inf'),0))[0])
    base_rank=next((i+1 for i,(p,_) in enumerate(rk) if p==BASE), None)
    print(f"  baseline {BASE} rank por {RANK_SEG} {RANK_MET}: #{base_rank}")
    print(f"\n  --- TOP-10 por {RANK_SEG} {RANK_MET} (mean) ---")
    print(f"  {'#':>2} {'fusion':<11}{'norm':<11}{'tP@5':>8}{'tN@5':>8}{'hP@5':>8}{'hN@5':>8}{'oN@5':>8}")
    def g(c,seg,met): return c.get((seg,met),(float('nan'),0))[0]
    for i,((m,nm),c) in enumerate(rk[:10],1):
        print(f"  {i:>2} {m:<11}{nm:<11}{g(c,'tail','precision@5'):>8.4f}{g(c,'tail','ndcg@5'):>8.4f}"
              f"{g(c,'head','precision@5'):>8.4f}{g(c,'head','ndcg@5'):>8.4f}{g(c,'overall','ndcg@5'):>8.4f}")

    # ---- LaTeX: matriz fusão×norm de tail ndcg@5 ----
    No=norms_sorted(N); methods_by_rank=[m for (m,nm),_ in rk]  # ordem por aparição no ranking
    seen=set(); Mo=[m for m in methods_by_rank if not (m in seen or seen.add(m))]
    tag=name.lower().replace("-","").replace(" ","")
    print(f"\n  --- LATEX matriz tail nDCG@5 ({name}) ---")
    print("\\begin{table*}[t]\\centering")
    print(f"\\caption{{{name}: tail nDCG@5 for all fusion$\\times$normalization combinations "
          f"({foldnote}; mean over folds). Baseline CombMNZ+ZMUV row in \\textbf{{bold}}. "
          f"Methods ordered by their best tail nDCG@5.}}")
    print(f"\\label{{tab:grid-{tag}}}")
    print("\\begin{tabular}{l"+"c"*len(No)+"}")
    print("\\toprule")
    print("Fusion & "+" & ".join(No)+" \\\\")
    print("\\midrule")
    for m in Mo:
        cellrow=[]
        for nm in No:
            v=cells.get((m,nm),{}).get(("tail","ndcg@5"))
            cellrow.append(f"{v[0]:.4f}" if v else "--")
        line=f"{m} & "+" & ".join(cellrow)+" \\\\"
        if m==BASE[0]: line="\\textbf{"+m+"} & "+" & ".join(cellrow)+" \\\\"
        print(line)
    print("\\bottomrule\\end{tabular}\\end{table*}")

    # ---- LaTeX: top-10 segmentado (cabeça/cauda + propensão) ----
    def gs(c,seg,met):
        v=c.get((seg,met)); return v if v else (float('nan'),0.0)
    def f4(x): return "--" if x!=x else f"{x:.4f}"   # x!=x detecta NaN (ex.: PSP ausente)
    def seg_row(label, nm, c):
        """Linha: Head(P@5,nDCG@5) | Tail(P@5,nDCG@5±std) | Propensity(PSP@5,PSnDCG@5).
        PSP/PSnDCG vêm do segmento overall (são propensity-scored = já ponderam a cauda)."""
        tn=gs(c,'tail','ndcg@5')
        tn_s=f"{f4(tn[0])}{{\\scriptsize$\\pm${tn[1]:.4f}}}" if tn[0]==tn[0] else "--"
        return (f"{label} & {nm} & {f4(gs(c,'head','precision@5')[0])} & {f4(gs(c,'head','ndcg@5')[0])}"
                f" & {f4(gs(c,'tail','precision@5')[0])} & {tn_s}"
                f" & {f4(gs(c,'overall','psp@5')[0])} & {f4(gs(c,'overall','psndcg@5')[0])} \\\\")
    print(f"\n  --- LATEX top-10 segmentado ({name}) ---")
    print("\\begin{table*}[t]\\centering\\small")
    print(f"\\caption{{{name}: top-10 fusion$\\times$normalization pairs by tail nDCG@5 "
          f"({foldnote}; mean$\\pm$std over folds for tail nDCG@5, mean otherwise). "
          f"PSP@5/PSnDCG@5 are propensity-scored (Jain et al.\\ 2016); ``--'' = not computed. "
          f"$\\dagger$ = baseline CombMNZ+ZMUV.}}")
    print(f"\\label{{tab:top-{tag}}}")
    print("\\begin{tabular}{llcccccc}")
    print("\\toprule")
    print("& & \\multicolumn{2}{c}{Head} & \\multicolumn{2}{c}{Tail} & \\multicolumn{2}{c}{Propensity} \\\\")
    print("\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-8}")
    print("Fusion & Norm & P@5 & nDCG@5 & P@5 & nDCG@5 & PSP@5 & PSnDCG@5 \\\\")
    print("\\midrule")
    for (m,nm),c in rk[:10]:
        dag="$\\dagger$" if (m,nm)==BASE else ""
        print(seg_row(f"{m}{dag}", nm, c))
    # se o baseline não estiver no top-10, anexa marcado
    if base_rank and base_rank>10:
        print("\\midrule")
        print(seg_row(f"{BASE[0]}$\\dagger$", BASE[1], cells[BASE])+f"  % rank #{base_rank}")
    if base_cells:                                   # baselines isolados (sem fusão) como referência
        print("\\midrule")
        for iso in ("dense","sparse"):
            c=base_cells.get((iso,"none"))
            if not c: continue
            print(seg_row(f"{iso} (no fusion)", "---", c))
    print("\\bottomrule\\end{tabular}\\end{table*}")

    # ---- LaTeX: significância pareada (se houver) ----
    if os.path.exists(spath):
        import csv as _csv
        srows=list(_csv.DictReader(open(spath)))
        print(f"\n  --- LATEX significância ({name}) ---")
        print("\\begin{table}[t]\\centering\\small")
        print(f"\\caption{{{name}: paired significance on tail nDCG@5 (Wilcoxon signed-rank; "
              f"$\\Delta$mean with 95\\% bootstrap CI; {foldnote}). $^{{*}}$ $p<0.05$.}}")
        print(f"\\label{{tab:sig-{tag}}}")
        print("\\begin{tabular}{lccr}")
        print("\\toprule")
        print("Comparison (tail nDCG@5) & $\\Delta$ mean & 95\\% CI & $p$ \\\\")
        print("\\midrule")
        for r in srows:
            p=float(r["p"]); star="$^{*}$" if p<0.05 else ""
            comp=(r["comparison"].replace("vs ","vs.\\ ").replace("denso","dense")
                  .replace("esparso","sparse").replace("ganho da fusão","fusion gain")
                  .replace("_","\\_"))
            print(f"{comp} & {float(r['delta']):+.4f}{star} & "
                  f"[{float(r['ci_lo']):+.4f}, {float(r['ci_hi']):+.4f}] & {p:.1e} \\\\")
        print("\\bottomrule\\end{tabular}\\end{table}")
    print()
