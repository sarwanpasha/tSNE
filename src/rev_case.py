"""Case studies for the revision.
 (1) Pango hierarchy: does the map place a descendant lineage next to its parent lineage?
     For each child->parent pair we report the rank of the parent among all lineage centroids
     ordered by distance from the child's centroid (1 = closest), for every method, embedding and seed.
 (2) Kernel disagreement as a quality flag: Spearman correlation between the mean neighborhood
     reliability of a sequence and its number of ambiguous characters, and enrichment of ambiguous
     sequences among the 5% least reliable points."""
import os, glob, numpy as np, pandas as pd
from scipy.stats import spearmanr
from rev_common import *
PAIRS = {'CoVNCBI': [('BQ.1.1', 'BA.5'), ('AY.4', 'B.1.617.2')], 'Spike7k': [('AY.4', 'B.1.617.2')]}
rows = []
for ds, pairs in PAIRS.items():
    y = labels(ds); labs = np.unique(y)
    for emb in EMBS:
        for fy in sorted(glob.glob(f'{R}/results/{ds}/{emb}/*_s[0-9]_Y.npy')):
            meth, seed = os.path.basename(fy)[:-6].rsplit('_s', 1); Y = np.load(fy)
            C = {l: Y[y == l].mean(0) for l in labs}
            for ch, pa in pairs:
                if ch not in C or pa not in C: continue
                dist = sorted(((np.linalg.norm(C[ch] - C[l]), l) for l in labs if l != ch))
                rank = [l for _, l in dist].index(pa) + 1
                rows.append(dict(ds=ds, emb=emb, method=meth, seed=int(seed), child=ch, parent=pa, parent_rank=rank, n_lineages=len(labs)))
pd.DataFrame(rows).to_csv(f'{R}/tables/case_hierarchy.csv', index=False)
rows = []
for ds in DATASETS:
    d = np.load(f'{R}/data/{ds}.npz', allow_pickle=True); seqs = [str(s).upper() for s in d['seqs']]; alpha = str(d['alpha'])
    std = set('ACGT') if lib.is_nt(alpha) else set('ACDEFGHIKLMNPQRSTVWY')
    amb = np.array([sum(1 for c in s if c not in std) for s in seqs])
    for emb in EMBS:
        f = f'{R}/results/{ds}/{emb}/consensus_reliability.npy'
        if not os.path.exists(f): continue
        rel = np.load(f).mean(0); low = rel <= np.quantile(rel, 0.05)
        rho, p = spearmanr(rel, amb) if amb.std() > 0 else (np.nan, np.nan)
        rows.append(dict(ds=ds, emb=emb, frac_amb_all=float((amb > 0).mean()), frac_amb_low5=float((amb[low] > 0).mean()),
                         mean_amb_low5=float(amb[low].mean()), mean_amb_all=float(amb.mean()), spearman_rel_amb=rho, p=p))
pd.DataFrame(rows).to_csv(f'{R}/tables/case_quality.csv', index=False); print('CASE_DONE', flush=True)
