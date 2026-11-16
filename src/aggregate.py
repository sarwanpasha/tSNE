"""Aggregate per-seed JSON results into summary CSVs (mean +/- std over seeds) + Friedman/Nemenyi ranks."""
import os, sys, glob, json, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import lib
R = lib.ROOT; rows = []
for f in glob.glob(f'{R}/results/*/*/*_s[0-9].json'):
    r = json.load(open(f)); base = {k: r.get(k) for k in ['ds', 'emb', 'method', 'seed', 'n', 'dim', 'tsne_sec', 'kernel_sec', 'auc_euclid', 'auc_kernel']}
    for c, v in (r.get('clf') or {}).items():
        for m, x in v.items(): base[f'{c}_{m}'] = x
    for m, x in (r.get('clu') or {}).items(): base[f'clu_{m}'] = x
    rows.append(base)
df = pd.DataFrame(rows); os.makedirs(f'{R}/tables', exist_ok=True)
df.to_csv(f'{R}/tables/all_runs.csv', index=False)
num = [c for c in df.columns if c not in ('ds', 'emb', 'method', 'seed')]
g = df.groupby(['ds', 'emb', 'method'])[num]
summ = g.mean().add_suffix('_mean').join(g.std().add_suffix('_std')).join(g.size().rename('nseeds'))
summ.reset_index().to_csv(f'{R}/tables/summary.csv', index=False)
# ranks per (ds,emb) block on auc_euclid / auc_kernel / KNN accuracy
from scipy.stats import friedmanchisquare
for metric in ['auc_euclid', 'auc_kernel', 'KNN_acc', 'clu_ari']:
    if metric not in df.columns: continue
    p = df.groupby(['ds', 'emb', 'method'])[metric].mean().unstack('method').dropna(axis=1, how='any')
    if p.shape[0] < 3 or p.shape[1] < 3: continue
    rk = p.rank(axis=1, ascending=False); stat, pv = friedmanchisquare(*[p[c] for c in p.columns])
    k, N = p.shape[1], p.shape[0]; q = {3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164, 11: 3.219, 12: 3.268, 13: 3.313, 14: 3.354, 15: 3.391, 16: 3.426, 17: 3.458, 18: 3.489, 19: 3.517, 20: 3.544}[k]
    cd = q * np.sqrt(k * (k + 1) / (6 * N))
    out = rk.mean().sort_values().rename('mean_rank').to_frame(); out['friedman_p'] = pv; out['nemenyi_CD'] = cd; out['blocks'] = N
    out.to_csv(f'{R}/tables/ranks_{metric}.csv'); print(metric, f'p={pv:.2e} CD={cd:.2f} N={N}'); print(out['mean_rank'].round(2).to_string())
print(summ.reset_index()[['ds', 'emb', 'method', 'auc_euclid_mean', 'auc_kernel_mean', 'nseeds']].to_string())
# paired Wilcoxon signed-rank: reference method (default NC-MKtSNE) vs every other method across (ds, emb) blocks, Holm-corrected
REF = sys.argv[1] if len(sys.argv) > 1 else 'NC-MKtSNE'
from scipy.stats import wilcoxon
for metric in ['auc_euclid', 'auc_kernel', 'KNN_acc', 'clu_ari']:
    if metric not in df.columns: continue
    p = df.groupby(['ds', 'emb', 'method'])[metric].mean().unstack('method')
    if REF not in p: continue
    res = []
    for m in p.columns:
        if m == REF: continue
        q = p[[REF, m]].dropna()
        if len(q) < 5: continue
        st = wilcoxon(q[REF], q[m]); res.append(dict(method=m, blocks=len(q), ref_wins=int((q[REF] > q[m]).sum()), median_diff=float((q[REF] - q[m]).median()), p=st.pvalue))
    if not res: continue
    r = pd.DataFrame(res).sort_values('p'); k = len(r); r['p_holm'] = np.minimum(1, np.maximum.accumulate(r['p'].values * (k - np.arange(k))))
    r.to_csv(f'{R}/tables/wilcoxon_{REF}_{metric}.csv', index=False); print('Wilcoxon', REF, 'vs others on', metric); print(r.round(4).to_string(index=False))
