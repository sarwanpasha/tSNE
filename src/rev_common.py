"""Shared helpers for the revision experiments (reuse cached per-kernel neighbor graphs)."""
import os, sys, json, time, glob, numpy as np, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib
R = lib.ROOT
NK = len(lib.KERNELS)

def load_block(ds, emb):
    z = np.load(f'{R}/data/cpk/{ds}_{emb}.npz')
    return z['IDX'], z['PP'], z['kt']

def labels(ds):
    d = np.load(f'{R}/data/{ds}.npz', allow_pickle=True); return np.asarray(d['labels']).astype(str)

def X_of(ds, emb):
    return sp.load_npz(f'{R}/data/emb/{ds}_{emb}.npz')

def hd_euclid(ds, emb):
    f = f'{R}/data/cpk/{ds}_{emb}_hdE.npy'
    if os.path.exists(f): return np.load(f)
    G = lib.gram(X_of(ds, emb)); De = np.sqrt(lib.sqdist_from_gram(G)); del G
    h = lib.knn_from_dist(De); tmp = f'{f}.{os.getpid()}.npy'; np.save(tmp, h); os.replace(tmp, f); return h

def jaccard_rows(A, B):
    k = A.shape[1]; out = np.empty(A.shape[0])
    for i in range(A.shape[0]):
        c = len(np.intersect1d(A[i], B[i], assume_unique=True)); out[i] = c / (2 * k - c)
    return out

def reliability(IDX, ks, kr):
    m = len(ks); A = np.zeros((m, IDX.shape[1]))
    for a, p in enumerate(ks):
        for q in ks:
            if p != q: A[a] += jaccard_rows(IDX[p][:, :kr], IDX[q][:, :kr]) / (m - 1)
    return A

def weights(A, tau, mode='point'):
    if mode == 'uniform': return np.full(A.shape, 1 / A.shape[0])
    if mode == 'global':
        g = A.mean(1); w = np.exp((g - g.max()) / tau); w /= w.sum(); return np.repeat(w[:, None], A.shape[1], 1)
    W = np.exp((A - A.max(0)) / tau); return W / W.sum(0)

def joint(IDX, PP, ks, W):
    n, K = PP.shape[1], PP.shape[2]; rows, cols, vals = [], [], []
    for a, p in enumerate(ks):
        rows.append(np.repeat(np.arange(n), K)); cols.append(IDX[p].ravel()); vals.append((PP[p] * W[a][:, None]).ravel())
    C = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)); C.sum_duplicates()
    J = C + C.T; return (J / J.sum()).tocsr()

def pca_init(X, seed):
    from sklearn.decomposition import TruncatedSVD
    from openTSNE import initialization
    Y = TruncatedSVD(2, random_state=seed).fit_transform(X)
    return initialization.rescale(np.ascontiguousarray(Y, dtype=np.float64))

def run_tsne(J, seed, threads, init=None):
    from openTSNE import TSNE, initialization, affinity
    n = J.shape[0]; aff = affinity.PrecomputedAffinities(J)
    if init is None: init = initialization.random(n, random_state=seed)
    t0 = time.time(); Y = np.asarray(TSNE(n_jobs=threads, random_state=seed).fit(affinities=aff, initialization=init)); return Y, time.time() - t0

def evaluate_and_save(ds, emb, name, seed, Y, hdE, y, extra):
    out = f'{R}/results/{ds}/{emb}'; os.makedirs(out, exist_ok=True)
    aE, RE = lib.auc_rnx(hdE, lib.knn_ld(Y))
    r = dict(ds=ds, emb=emb, method=name, seed=seed, n=len(y), auc_euclid=aE, R_euclid=RE.tolist(),
             clf=lib.classify(Y, y, seed), clu=lib.cluster(Y, y, seed), **extra)
    np.save(f'{out}/{name}_s{seed}_Y.npy', Y.astype(np.float32)); json.dump(r, open(f'{out}/{name}_s{seed}.json', 'w'))
    print(ds, emb, name, seed, f'aucE={aE:.4f} knn={r["clf"]["KNN"]["acc"]:.4f}', flush=True)

def done(ds, emb, name, seed):
    return os.path.exists(f'{R}/results/{ds}/{emb}/{name}_s{seed}.json')

def cond_from_knn(idx, d, perp):
    """perplexity-calibrated conditionals from sorted kNN distances (n x K)."""
    n = d.shape[0]; d2 = d ** 2; d2 = d2 - d2[:, :1]
    lo = np.full(n, 1e-20); hi = np.full(n, 1e20); beta = np.ones(n); H0 = np.log(perp)
    for _ in range(100):
        P = np.exp(-d2 * beta[:, None]); P /= P.sum(1, keepdims=True)
        H = -(P * np.log(np.maximum(P, 1e-300))).sum(1)
        big = H > H0; lo = np.where(big, beta, lo); hi = np.where(big, hi, beta)
        beta = np.where(hi >= 1e19, beta * 2, (lo + hi) / 2)
    return P

DATASETS = ['Spike7k', 'Host', 'ShortRead', 'Rabies', 'Genome', 'BreastCancer', 'LungCancer', 'Dengue', 'InfluenzaHA', 'CoVNCBI']
EMBS = ['OHE', 'Spike2Vec', 'Minimizer', 'SpacedKmer']
