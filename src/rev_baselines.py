"""Additional baselines for the revision.
  multiscale : multiscale SNE affinities (Lee et al. 2015) on Euclidean distances, perplexities 4..64
  metavis    : spectral meta-visualization (Ma, Sun & Zou 2023) combining the 9 single-kernel t-SNE maps
usage: rev_baselines.py MODE DS EMB [--seeds 5] [--threads 8]"""
import argparse, os
ap = argparse.ArgumentParser(); ap.add_argument('mode'); ap.add_argument('ds'); ap.add_argument('emb')
ap.add_argument('--seeds', type=int, default=5); ap.add_argument('--threads', type=int, default=8)
a = ap.parse_args()
for v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS']: os.environ[v] = str(a.threads)
import numpy as np, scipy.sparse as sp
from rev_common import *
y = labels(a.ds); hdE = hd_euclid(a.ds, a.emb); n = len(y)

def to_joint(idx, P):
    K = P.shape[1]; C = sp.csr_matrix((P.ravel(), (np.repeat(np.arange(n), K), idx.ravel())), shape=(n, n)); C.sum_duplicates()
    J = C + C.T; return (J / J.sum()).tocsr()

if a.mode == 'multiscale':
    perps = [4, 8, 16, 32, 64]; K = min(3 * max(perps), n - 1)
    G = lib.gram(X_of(a.ds, a.emb)); D = np.sqrt(lib.sqdist_from_gram(G)); del G; np.fill_diagonal(D, np.inf)
    idx = np.argpartition(D, K, axis=1)[:, :K]; d = np.take_along_axis(D, idx, 1); del D
    o = np.argsort(d, 1); idx = np.take_along_axis(idx, o, 1); d = np.take_along_axis(d, o, 1)
    P = np.mean([cond_from_knn(idx, d, p) for p in perps], 0); J = to_joint(idx, P)
    for s in range(a.seeds):
        if done(a.ds, a.emb, 'MultiscaleSNE', s): continue
        Y, t = run_tsne(J, s, a.threads); evaluate_and_save(a.ds, a.emb, 'MultiscaleSNE', s, Y, hdE, y, dict(tsne_sec=t, perplexities=perps))

elif a.mode == 'metavis':
    from scipy.spatial.distance import cdist
    from openTSNE import TSNE, affinity, initialization
    for s in range(a.seeds):
        if done(a.ds, a.emb, 'MetaVis', s): continue
        Ds = []
        for k in lib.KERNELS:
            Y = np.load(f'{R}/results/{a.ds}/{a.emb}/{k}_s{s}_Y.npy').astype(np.float64)
            D = cdist(Y, Y).astype(np.float32); D /= np.linalg.norm(D, axis=1, keepdims=True) + 1e-12; Ds.append(D)
        m = len(Ds); C = np.zeros((n, m, m), np.float32)
        for p in range(m):
            for q in range(p, m):
                c = np.einsum('ij,ij->i', Ds[p], Ds[q]); C[:, p, q] = c; C[:, q, p] = c
        w, V = np.linalg.eigh(C); sc = np.abs(V[:, :, -1])          # eigenscores per point and visualization
        sc = sc / sc.sum(1, keepdims=True)
        M = np.zeros((n, n), np.float32)
        for p in range(m): M += sc[:, p:p + 1] * Ds[p]
        del Ds; M = (M + M.T) / 2; np.fill_diagonal(M, 0)
        aff = affinity.PerplexityBasedNN(np.ascontiguousarray(M, dtype=np.float64), perplexity=30, metric='precomputed', n_jobs=a.threads, random_state=s)
        import time; t0 = time.time()
        Y = np.asarray(TSNE(n_jobs=a.threads, random_state=s).fit(affinities=aff, initialization=initialization.random(n, random_state=s)))
        evaluate_and_save(a.ds, a.emb, 'MetaVis', s, Y, hdE, y, dict(tsne_sec=time.time() - t0, eigenscore_mean=sc.mean(0).round(4).tolist()))
print('BASE_DONE', a.mode, a.ds, a.emb, flush=True)
