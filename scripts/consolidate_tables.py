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

def load(path):
    cells = defaultdict(dict)  # (method,norm) -> {(seg,met):(mean,std)}
    M,N,S,K = set(),set(),set(),set()
    n=0
    with open(path) as fh:
        for r in csv.DictReader(fh):
            cells[(r["method"],r["norm"])][(r["segment"],r["metric"])]=(float(r["mean"]),float(r["std"]))
            M.add(r["method"]); N.add(r["norm"]); S.add(r["segment"]); K.add(r["metric"]); n+=1
    return cells,M,N,S,K,n

def norms_sorted(N):
    pref=[x for x in NORM_ORDER if x in N]; return pref+sorted(N-set(pref))

for name,path,foldnote in DATASETS:
    print("="*70); print(f"# {name}  ({path})")
    if not os.path.exists(path): print("  AUSENTE"); continue
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

    # ---- LaTeX: top-10 segmentado (cabeça/cauda) ----
    def gs(c,seg,met):
        v=c.get((seg,met)); return v if v else (float('nan'),0.0)
    print(f"\n  --- LATEX top-10 segmentado ({name}) ---")
    print("\\begin{table}[t]\\centering\\small")
    print(f"\\caption{{{name}: top-10 fusion$\\times$normalization pairs by tail nDCG@5 "
          f"({foldnote}; mean$\\pm$std over folds for tail nDCG@5, mean otherwise). "
          f"$\\dagger$ = baseline CombMNZ+ZMUV.}}")
    print(f"\\label{{tab:top-{tag}}}")
    print("\\begin{tabular}{llcccc}")
    print("\\toprule")
    print("& & \\multicolumn{2}{c}{Head} & \\multicolumn{2}{c}{Tail} \\\\")
    print("\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}")
    print("Fusion & Norm & P@5 & nDCG@5 & P@5 & nDCG@5 \\\\")
    print("\\midrule")
    for (m,nm),c in rk[:10]:
        dag="$\\dagger$" if (m,nm)==BASE else ""
        tn=gs(c,'tail','ndcg@5')
        print(f"{m}{dag} & {nm} & {gs(c,'head','precision@5')[0]:.4f} & {gs(c,'head','ndcg@5')[0]:.4f}"
              f" & {gs(c,'tail','precision@5')[0]:.4f} & {tn[0]:.4f}{{\\scriptsize$\\pm${tn[1]:.4f}}} \\\\")
    # se o baseline não estiver no top-10, anexa marcado
    if base_rank and base_rank>10:
        (m,nm)=BASE; c=cells[BASE]; tn=gs(c,'tail','ndcg@5')
        print("\\midrule")
        print(f"{m}$\\dagger$ & {nm} & {gs(c,'head','precision@5')[0]:.4f} & {gs(c,'head','ndcg@5')[0]:.4f}"
              f" & {gs(c,'tail','precision@5')[0]:.4f} & {tn[0]:.4f}{{\\scriptsize$\\pm${tn[1]:.4f}}} \\\\"
              f"  % rank #{base_rank}")
    if base_cells:                                   # baselines isolados (sem fusão) como referência
        print("\\midrule")
        for iso in ("dense","sparse"):
            c=base_cells.get((iso,"none"))
            if not c: continue
            tn=gs(c,'tail','ndcg@5')
            print(f"{iso} (no fusion) & --- & {gs(c,'head','precision@5')[0]:.4f} & {gs(c,'head','ndcg@5')[0]:.4f}"
                  f" & {gs(c,'tail','precision@5')[0]:.4f} & {tn[0]:.4f}{{\\scriptsize$\\pm${tn[1]:.4f}}} \\\\")
    print("\\bottomrule\\end{tabular}\\end{table}")

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
