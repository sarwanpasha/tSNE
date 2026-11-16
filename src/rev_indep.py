"""Neighborhood preservation against references that do not depend on any kernel.
Protein/peptide data: normalized Levenshtein (edit) distance. Nucleotide data: Mash distance from
MinHash sketches (k=21, s=1000). Computed on a random subsample of m sequences; every stored 2D map
(all methods, embeddings, seeds) is scored with AUC_RNX against these reference neighbors.
Also scores each kernel's own high-dimensional neighbors against the reference.
usage: rev_indep.py DS [--m 2000] [--threads 16]"""
import argparse, os, glob
ap = argparse.ArgumentParser(); ap.add_argument('ds'); ap.add_argument('--m', type=int, default=2000); ap.add_argument('--threads', type=int, default=16)
a = ap.parse_args()
for v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS']: os.environ[v] = str(a.threads)
import numpy as np, pandas as pd
from rev_common import *
d = np.load(f'{R}/data/{a.ds}.npz', allow_pickle=True); seqs = [str(s) for s in d['seqs']]; alpha = str(d['alpha'])
n = len(seqs); m = min(a.m, n); sub = np.sort(np.random.default_rng(0).choice(n, m, replace=False))
S = [seqs[i] for i in sub]; nt = lib.is_nt(alpha)
f = f'{R}/data/cpk/{a.ds}_refdist_{m}.npy'
if os.path.exists(f): Dref = np.load(f)
elif not nt:
    from rapidfuzz.process import cdist
    from rapidfuzz.distance import Levenshtein
    Dref = cdist(S, S, scorer=Levenshtein.normalized_distance, workers=a.threads).astype(np.float64); np.save(f, Dref)
else:
    import hashlib
    k, sz = 21, 1000; sk = []
    for s in S:
        h = {int.from_bytes(hashlib.blake2b(s[i:i + k].encode(), digest_size=8).digest(), 'little') for i in range(len(s) - k + 1) if 'N' not in s[i:i + k]}
        sk.append(np.array(sorted(h)[:sz], dtype=np.uint64))
    Dref = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1, m):
            u = np.union1d(sk[i], sk[j])[:sz]; inter = np.intersect1d(np.intersect1d(sk[i], sk[j], assume_unique=True), u, assume_unique=True).size
            J = inter / max(len(u), 1); Dref[i, j] = Dref[j, i] = 1.0 if J == 0 else -np.log(2 * J / (1 + J)) / k
    np.save(f, Dref)
kk = min(99, m - 2); hdR = lib.knn_from_dist(Dref, k=kk)
rows = []
for emb in EMBS:
    for fy in sorted(glob.glob(f'{R}/results/{a.ds}/{emb}/*_s[0-9]_Y.npy')):
        name = os.path.basename(fy)[:-6]; meth, seed = name.rsplit('_s', 1)
        Y = np.load(fy)[sub]; auc, _ = lib.auc_rnx(hdR, lib.knn_ld(Y, k=kk))
        rows.append(dict(ds=a.ds, emb=emb, method=meth, seed=int(seed), auc_ref=auc))
    X = X_of(a.ds, emb)[sub]
    for kname in lib.KERNELS:
        Kk, _ = lib.kernel(kname, X); Dk = lib.kernel_dist(Kk)
        auc, _ = lib.auc_rnx(hdR, lib.knn_from_dist(Dk, k=kk)); rows.append(dict(ds=a.ds, emb=emb, method='HD:' + kname, seed=-1, auc_ref=auc))
os.makedirs(f'{R}/tables', exist_ok=True); pd.DataFrame(rows).to_csv(f'{R}/tables/indep_{a.ds}.csv', index=False)
print('INDEP_DONE', a.ds, len(rows), 'nt' if nt else 'aa', flush=True)
