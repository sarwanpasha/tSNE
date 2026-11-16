"""Statistics for the revision: results with and without OHE, dataset-level tests (embeddings averaged
within a dataset, so blocks are independent), paired effect sizes with dataset-level bootstrap CIs,
and summaries of the sensitivity, baseline and independent-reference experiments."""
import os, glob, json, numpy as np, pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from rev_common import R
rows = []
for f in glob.glob(f'{R}/results/*/*/*_s[0-9].json'):
    r = json.load(open(f))
    rows.append(dict(ds=r['ds'], emb=r['emb'], method=r['method'], seed=r['seed'], aucE=r.get('auc_euclid'),
                     knn=r.get('clf', {}).get('KNN', {}).get('acc'), ari=r.get('clu', {}).get('ari')))
df = pd.DataFrame(rows); df.to_csv(f'{R}/tables/rev_all_runs.csv', index=False)
blk = df.groupby(['ds', 'emb', 'method'])[['aucE', 'knn', 'ari']].mean().reset_index()
MAIN = ['NC-MKtSNE', 'NC-global', 'MK-mean', 'Gaussian', 'Linear', 'Cosine', 'Laplacian', 'ChiSquared', 'AdditiveChi2', 'Polynomial',
        'Sigmoid', 'Isolation', 'UMAP-Euclid', 'UMAP-Cosine', 'NC-dedup', 'MK-mean-dedup', 'm-SNE', 'MultiscaleSNE', 'MetaVis',
        'NC-MKtSNE-PCA', 'Gaussian-PCA', 'Linear-PCA']
out = {}
def ranks(b, metric, methods):
    t = b[b.method.isin(methods)].pivot_table(index=['ds', 'emb'] if 'emb' in b else 'ds', columns='method', values=metric).dropna()
    rk = t.rank(axis=1, ascending=False).mean().sort_values(); p = friedmanchisquare(*[t[c] for c in t.columns]).pvalue if t.shape[1] > 2 else np.nan
    return rk.round(2).to_dict(), float(p), int(t.shape[0])
for metric in ['aucE', 'knn', 'ari']:
    avail = [m for m in MAIN if m in set(blk.method)]
    full = blk.pivot_table(index=['ds', 'emb'], columns='method', values=metric)
    common = [m for m in avail if full[m].notna().sum() == full.shape[0]]
    out[f'{metric}_ranks_all'] = ranks(blk, metric, common)
    out[f'{metric}_ranks_noOHE'] = ranks(blk[blk.emb != 'OHE'], metric, common)
    dsl = blk[blk.emb != 'OHE'].groupby(['ds', 'method'])[metric].mean().reset_index()
    out[f'{metric}_ranks_dataset_noOHE'] = ranks(dsl, metric, common)
    tab = dsl.pivot_table(index='ds', columns='method', values=metric); eff = {}
    rng = np.random.default_rng(0)
    for m in common:
        if m == 'NC-MKtSNE': continue
        dlt = (tab['NC-MKtSNE'] - tab[m]).dropna().values
        bs = [rng.choice(dlt, len(dlt)).mean() for _ in range(5000)]
        try: p = wilcoxon(dlt).pvalue
        except ValueError: p = 1.0
        eff[m] = dict(mean_diff=float(dlt.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], wins=int((dlt > 0).sum()), n=len(dlt), p=float(p))
    out[f'{metric}_effects_dataset_noOHE'] = eff
sens = blk[blk.method.str.match(r'NC-(tau|kr)') | (blk.method == 'NC-MKtSNE') | (blk.method == 'MK-mean')]
out['sensitivity'] = sens[sens.emb != 'OHE'].groupby('method')[['aucE', 'knn']].mean().round(4).to_dict()
ind = pd.concat([pd.read_csv(f) for f in glob.glob(f'{R}/tables/indep_*.csv')]) if glob.glob(f'{R}/tables/indep_*.csv') else None
if ind is not None:
    ib = ind[ind.seed >= 0].groupby(['ds', 'emb', 'method']).auc_ref.mean().reset_index()
    out['indep_ranks_noOHE'] = ranks(ib[ib.emb != 'OHE'].rename(columns={'auc_ref': 'v'}), 'v', [m for m in MAIN if m in set(ib.method)])
    out['indep_dataset_means_noOHE'] = ib[ib.emb != 'OHE'].groupby(['method']).auc_ref.mean().sort_values(ascending=False).round(4).to_dict()
    out['indep_HD_kernels'] = ind[ind.seed < 0].groupby('method').auc_ref.mean().round(4).to_dict()
    ib.pivot_table(index='ds', columns='method', values='auc_ref').round(4).to_csv(f'{R}/tables/indep_by_dataset.csv')
for f in ['case_hierarchy', 'case_quality']:
    if os.path.exists(f'{R}/tables/{f}.csv'):
        c = pd.read_csv(f'{R}/tables/{f}.csv')
        out[f] = (c[c.emb == 'Spike2Vec'].groupby(['ds', 'child', 'method']).parent_rank.mean().round(2).unstack(0).to_dict() if f == 'case_hierarchy'
                  else c.round(4).to_dict(orient='records'))
def sk(o): return {str(k): sk(v) for k, v in o.items()} if isinstance(o, dict) else o
json.dump(sk(out), open(f'{R}/tables/rev_stats.json', 'w'), indent=1, default=str); print('STATS_DONE')
