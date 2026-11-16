"""Revision experiments on the consensus (reuse cached neighbor graphs):
  dedup  : consensus over a de-duplicated kernel set (drops Linear and AdditiveChi2)
  tau    : temperature sensitivity      kr : neighborhood size of the reliability score
  pca    : PCA initialization for NC-MKt-SNE, Gaussian and Euclidean (linear) t-SNE
  msne   : m-SNE style alternating global weights (init: MK-mean embedding)
usage: rev_consensus.py MODE DS EMB [--seeds 5] [--threads 8]"""
import argparse, os
ap = argparse.ArgumentParser(); ap.add_argument('mode'); ap.add_argument('ds'); ap.add_argument('emb')
ap.add_argument('--seeds', type=int, default=5); ap.add_argument('--threads', type=int, default=8)
a = ap.parse_args()
for v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS']: os.environ[v] = str(a.threads)
import numpy as np
from rev_common import *
IDX, PP, kt = load_block(a.ds, a.emb); y = labels(a.ds); hdE = hd_euclid(a.ds, a.emb)
ALL = list(range(NK)); K = lib.KERNELS
DEDUP = [i for i, k in enumerate(K) if k not in ('Linear', 'AdditiveChi2')]

def variant(name, ks, tau=0.05, kr=10, mode='point', init='random', seeds=a.seeds):
    A = reliability(IDX, ks, kr); W = weights(A, tau, mode); J = joint(IDX, PP, ks, W)
    for s in range(seeds):
        if done(a.ds, a.emb, name, s): continue
        ini = pca_init(X_of(a.ds, a.emb), s) if init == 'pca' else None
        Y, t = run_tsne(J, s, a.threads, ini)
        evaluate_and_save(a.ds, a.emb, name, s, Y, hdE, y, dict(tsne_sec=t, tau=tau, kr=kr, kernels=[K[i] for i in ks], wmean=W.mean(1).round(4).tolist()))

if a.mode == 'dedup':
    variant('NC-dedup', DEDUP); variant('MK-mean-dedup', DEDUP, mode='uniform')
elif a.mode == 'tau':
    for t in [0.01, 0.02, 0.1, 0.2, 0.5, 1.0]: variant(f'NC-tau{t}', ALL, tau=t, seeds=min(a.seeds, 3))
elif a.mode == 'kr':
    for k in [5, 20, 30, 60]: variant(f'NC-kr{k}', ALL, kr=k, seeds=min(a.seeds, 3))
elif a.mode == 'pca':
    variant('NC-MKtSNE-PCA', ALL, init='pca')
    for nm in ['Gaussian', 'Linear']:
        i = K.index(nm); variant(f'{nm}-PCA', [i], mode='uniform', init='pca')
elif a.mode == 'msne':
    # m-SNE (Xie et al. 2011): min_alpha sum_m alpha_m KL(P_m||Q) + lam*||alpha||^2 on the simplex, alternated with Y
    import scipy.sparse as sp
    Ps = [joint(IDX, PP, [m], np.ones((1, IDX.shape[1]))) for m in ALL]
    def kls(Y):
        n = Y.shape[0]; Z = 0.0
        for i0 in range(0, n, 2000):
            d2 = ((Y[i0:i0 + 2000, None, :] - Y[None, :, :]) ** 2).sum(-1); Z += (1 / (1 + d2)).sum() - min(2000, n - i0)
        out = []
        for P in Ps:
            P = P.tocoo(); m = P.row != P.col; p = P.data[m]
            d2 = ((Y[P.row[m]] - Y[P.col[m]]) ** 2).sum(1); lq = -np.log1p(d2) - np.log(Z)
            out.append(float((p * (np.log(p) - lq)).sum()))
        return np.array(out)
    def proj_simplex(v):
        u = np.sort(v)[::-1]; c = np.cumsum(u); r = np.nonzero(u * np.arange(1, len(v) + 1) > (c - 1))[0][-1]
        th = (c[r] - 1) / (r + 1); return np.maximum(v - th, 0)
    for s in range(a.seeds):
        if done(a.ds, a.emb, 'm-SNE', s): continue
        Y = np.load(f'{R}/results/{a.ds}/{a.emb}/MK-mean_s{s}_Y.npy').astype(np.float64); tt = 0
        for it in range(2):
            kl = kls(Y); lam = max(kl.max() - kl.min(), 1e-6); alpha = proj_simplex(-kl / (2 * lam))
            J = joint(IDX, PP, ALL, np.repeat(alpha[:, None], IDX.shape[1], 1))
            from openTSNE import initialization
            Y, t = run_tsne(J, s, a.threads, initialization.rescale(Y)); tt += t
        evaluate_and_save(a.ds, a.emb, 'm-SNE', s, Y, hdE, y, dict(tsne_sec=tt, alpha=alpha.round(4).tolist()))
print('REV_DONE', a.mode, a.ds, a.emb, flush=True)
