"""Compact text report of the revision experiments (dataset-level means without OHE, ranks, effect sizes,
scaling, validation and case studies). usage: rev_report.py [section]   sections: methods effects scale case all"""
import sys, glob, json, numpy as np, pandas as pd
from scipy.stats import wilcoxon
from rev_common import R
sec = sys.argv[1] if len(sys.argv) > 1 else 'all'
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
df = pd.read_csv(f'{R}/tables/rev_all_runs.csv'); df = df[df.emb != 'OHE']
ind = pd.concat([pd.read_csv(f) for f in glob.glob(f'{R}/tables/indep_*.csv') if 'by_dataset' not in f])
ind = ind[(ind.emb != 'OHE') & (ind.seed >= 0)].rename(columns={'auc_ref': 'indep'})
b = df.groupby(['ds', 'emb', 'method'])[['aucE', 'knn']].mean().reset_index()
b = b.merge(ind.groupby(['ds', 'emb', 'method']).indep.mean().reset_index(), how='left')
d = b.groupby(['ds', 'method'])[['aucE', 'indep', 'knn']].mean().reset_index()
ndb = b.groupby('method').size(); full = ndb[ndb == ndb.max()].index
d = d[d.method.isin(full)]
if sec in ('methods', 'all'):
    t = d.groupby('method')[['aucE', 'indep', 'knn']].mean()
    for m in ['aucE', 'indep', 'knn']:
        core = [x for x in t.index if not x.startswith(('NC-tau', 'NC-kr', 'Iso-psi', 'NC3', 'tSNE-'))]
        p = d[d.method.isin(core)].pivot_table(index='ds', columns='method', values=m)
        t.loc[core, 'rk_' + m] = p.rank(axis=1, ascending=False).mean()
    print(t.sort_values('rk_aucE').round(4).to_string())
if sec in ('effects', 'all'):
    for m in ['aucE', 'indep', 'knn']:
        p = d.pivot_table(index='ds', columns='method', values=m); out = []
        for c in p.columns:
            if c == 'NC-MKtSNE': continue
            x = (p['NC-MKtSNE'] - p[c]).dropna()
            try: pv = wilcoxon(x).pvalue
            except ValueError: pv = 1.0
            out.append((c, x.mean(), int((x > 0).sum()), len(x), pv))
        print(m); print(pd.DataFrame(out, columns=['vs', 'diff', 'wins', 'n', 'p']).sort_values('diff').round(4).to_string(index=False))
if sec in ('scale', 'all'):
    for l in open(f'{R}/tables/scale_fast.jsonl'):
        r = json.loads(l); print({k: round(v, 1) if isinstance(v, float) else v for k, v in r.items()})
    a = pd.read_csv(f'{R}/tables/rev_all_runs.csv'); v = a[a.method.isin(['NC3-fast', 'NC3-exact'])]
    print(v.groupby(['ds', 'method'])[['aucE', 'knn']].agg(['mean', 'std']).round(4).to_string())
if sec in ('case', 'all'):
    h = pd.read_csv(f'{R}/tables/case_hierarchy.csv'); h = h[h.emb != 'OHE']
    keep = ['NC-MKtSNE', 'NC-global', 'MK-mean', 'Gaussian', 'Linear', 'Cosine', 'Laplacian', 'Isolation', 'm-SNE', 'MetaVis', 'MultiscaleSNE', 'UMAP-Euclid']
    print(h[h.method.isin(keep)].pivot_table(index='method', columns=['ds', 'child'], values='parent_rank', aggfunc='mean').round(2).to_string())
    print(h.groupby(['ds', 'child']).n_lineages.first().to_string())
    q = pd.read_csv(f'{R}/tables/case_quality.csv'); print(q.round(4).to_string(index=False))
