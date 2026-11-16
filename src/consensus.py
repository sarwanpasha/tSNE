"""Neighbourhood-Consensus Multi-Kernel t-SNE (NC-MKt-SNE).
For each kernel k: kNN graph + perplexity-calibrated conditional P_k(j|i).
Per-point reliability a_{k,i} = mean_l Jaccard(N_k(i), N_l(i)); weights w_{k,i} = softmax_k(a_{k,i}/tau).
Consensus conditional P(j|i) = sum_k w_{k,i} P_k(j|i); symmetrize; run t-SNE on it.
Variants: 'NC-MKtSNE' (per-point weights), 'MK-mean' (uniform weights), 'NC-global' (per-kernel weights)."""
import os, sys, argparse
ap = argparse.ArgumentParser(); ap.add_argument('ds'); ap.add_argument('emb')
ap.add_argument('--seeds', type=int, default=5); ap.add_argument('--threads', type=int, default=8)
ap.add_argument('--tau', type=float, default=0.05); ap.add_argument('--perp', type=float, default=30)
a = ap.parse_args()
for v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS']: os.environ[v] = str(a.threads)
import json, time, numpy as np, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import lib

def cond_P(D, perp, K):
    n = D.shape[0]; D = D.copy(); np.fill_diagonal(D, np.inf)
    idx = np.argpartition(D, K, axis=1)[:, :K]; d = np.take_along_axis(D, idx, 1)
    o = np.argsort(d, 1); idx = np.take_along_axis(idx, o, 1); d = np.take_along_axis(d, o, 1)
    d2 = d ** 2; d2 = d2 - d2[:, :1]
    lo = np.full(n, 1e-20); hi = np.full(n, 1e20); beta = np.ones(n); H0 = np.log(perp)
    for _ in range(100):
        P = np.exp(-d2 * beta[:, None]); s = P.sum(1, keepdims=True); P /= s
        H = -(P * np.log(np.maximum(P, 1e-300))).sum(1)
        big = H > H0; lo = np.where(big, beta, lo); hi = np.where(big, hi, beta)
        beta = np.where(hi >= 1e19, beta * 2, (lo + hi) / 2)
    return idx, P

def jaccard_rows(A, B):  # A,B: n x k index arrays
    k = A.shape[1]; out = np.empty(A.shape[0])
    for i in range(A.shape[0]): out[i] = len(np.intersect1d(A[i], B[i], assume_unique=True)) / (2 * k - len(np.intersect1d(A[i], B[i], assume_unique=True)))
    return out

def to_joint(idx, P):
    n, K = P.shape
    C = sp.csr_matrix((P.ravel(), (np.repeat(np.arange(n), K), idx.ravel())), shape=(n, n))
    J = (C + C.T); J = J / J.sum(); return J.tocsr()

R = lib.ROOT; d = np.load(f'{R}/data/{a.ds}.npz', allow_pickle=True); y = np.asarray(d['labels']).astype(str)
X = sp.load_npz(f'{R}/data/emb/{a.ds}_{a.emb}.npz'); n = X.shape[0]; K = int(3 * a.perp)
out = f'{R}/results/{a.ds}/{a.emb}'; os.makedirs(out, exist_ok=True)
cache = f'{R}/data/cpk/{a.ds}_{a.emb}.npz'; os.makedirs(os.path.dirname(cache), exist_ok=True)
t0 = time.time()
if os.path.exists(cache):
    z = np.load(cache); IDX, PP, ktimes = z['IDX'], z['PP'], z['kt']
else:
    IDX, PP, ktimes = [], [], []
    for k in lib.KERNELS:
        Kk, kt = lib.kernel(k, X); Dk = lib.kernel_dist(Kk); del Kk
        i_, p_ = cond_P(Dk, a.perp, K); del Dk; IDX.append(i_); PP.append(p_); ktimes.append(kt); print(k, 'done', f'{kt:.1f}s', flush=True)
    IDX, PP, ktimes = np.stack(IDX), np.stack(PP), np.array(ktimes); np.savez(cache, IDX=IDX, PP=PP, kt=ktimes)
nk = len(lib.KERNELS); k10 = 10
A = np.zeros((nk, n))
for p in range(nk):
    for q in range(nk):
        if p != q: A[p] += jaccard_rows(IDX[p][:, :k10], IDX[q][:, :k10]) / (nk - 1)
np.save(f'{out}/consensus_reliability.npy', A.astype(np.float32))
print('mean reliability per kernel', dict(zip(lib.KERNELS, A.mean(1).round(3))), flush=True)
G = lib.gram(X); De = np.sqrt(lib.sqdist_from_gram(G)); del G; hdE = lib.knn_from_dist(De); del De

def build(variant):
    if variant == 'MK-mean': W = np.full((nk, n), 1 / nk)
    elif variant == 'NC-global': g = A.mean(1); W = np.repeat(np.exp((g - g.max()) / a.tau)[:, None], n, 1); W /= W.sum(0)
    else: W = np.exp((A - A.max(0)) / a.tau); W /= W.sum(0)
    rows, cols, vals = [], [], []
    for p in range(nk):
        rows.append(np.repeat(np.arange(n), K)); cols.append(IDX[p].ravel()); vals.append((PP[p] * W[p][:, None]).ravel())
    C = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)); C.sum_duplicates()
    J = C + C.T; J = (J / J.sum()).tocsr(); return J, W

from openTSNE import TSNE, initialization, affinity
for variant in ['NC-MKtSNE', 'MK-mean', 'NC-global']:
    J, W = build(variant); np.save(f'{out}/{variant}_weights.npy', W.mean(1).astype(np.float32))
    hdC = np.asarray([np.argsort(-J[i].toarray().ravel())[:99] for i in range(n)]) if n <= 30000 else None
    for s in range(a.seeds):
        f = f'{out}/{variant}_s{s}.json'
        if os.path.exists(f): continue
        t1 = time.time(); aff = affinity.PrecomputedAffinities(J)
        Y = np.asarray(TSNE(n_jobs=a.threads, random_state=s).fit(affinities=aff, initialization=initialization.random(n, random_state=s))); tt = time.time() - t1
        aE, RE = lib.auc_rnx(hdE, lib.knn_ld(Y)); aK, RK = lib.auc_rnx(hdC, lib.knn_ld(Y))
        r = dict(ds=a.ds, emb=a.emb, method=variant, seed=s, n=n, dim=X.shape[1], tsne_sec=tt, kernel_sec=float(ktimes.sum()),
                 auc_euclid=aE, auc_kernel=aK, R_euclid=RE.tolist(), R_kernel=RK.tolist(), weights=dict(zip(lib.KERNELS, W.mean(1).round(4).tolist())),
                 clf=lib.classify(Y, y, s), clu=lib.cluster(Y, y, s))
        np.save(f'{out}/{variant}_s{s}_Y.npy', Y.astype(np.float32)); json.dump(r, open(f, 'w'))
        print(variant, s, f'aucE={aE:.3f} aucK={aK:.3f} knn={r["clf"]["KNN"]["acc"]:.3f}', flush=True)
print('CONSENSUS_DONE', a.ds, a.emb, f'{time.time()-t0:.0f}s', flush=True)
