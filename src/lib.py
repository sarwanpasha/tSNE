import os, time, json, numpy as np, scipy.sparse as sp
from numba import njit, prange
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from scipy.ndimage import minimum_filter1d

ROOT = os.path.abspath(os.environ.get('TSNE_ROOT', 'workspace'))

# ---------------- alphabet / encoding ----------------
def clean(seqs, minfrac=1e-4):
    seqs = [str(s).upper().replace('-', '').replace('*', '').replace(' ', '').replace('"', '') for s in seqs]
    cnt = {}
    for s in seqs:
        for c in set(s): cnt[c] = cnt.get(c, 0) + s.count(c)
    tot = sum(cnt.values()); keep = sorted(c for c, v in cnt.items() if v / tot >= minfrac)
    rare = set(cnt) - set(keep)
    if rare:
        tr = str.maketrans({c: 'X' for c in rare}); seqs = [s.translate(tr) for s in seqs]
        if 'X' not in keep: keep = sorted(keep + ['X'])
    return seqs, ''.join(keep)

def codes(s, alpha):
    lut = np.full(256, 0, np.int64)
    for i, c in enumerate(alpha): lut[ord(c)] = i
    return lut[np.frombuffer(s.encode(), np.uint8)]

def is_nt(alpha):
    return set(alpha) <= set('ACGTUNX')

def drop_zero_cols(X):
    X = X.tocsc(); nz = np.diff(X.indptr) > 0
    return X[:, nz].tocsr().astype(np.float32)

# ---------------- embeddings ----------------
def emb_ohe(seqs, alpha):
    A = len(alpha); L = max(len(s) for s in seqs); rows, cols = [], []
    for i, s in enumerate(seqs):
        c = codes(s, alpha); rows.append(np.full(len(c), i)); cols.append(np.arange(len(c)) * A + c)
    r = np.concatenate(rows); c = np.concatenate(cols)
    return drop_zero_cols(sp.csr_matrix((np.ones(len(r), np.float32), (r, c)), shape=(len(seqs), L * A)))

def _int_kmers(c, k, A):
    n = len(c) - k + 1
    if n <= 0: return np.zeros(0, np.int64)
    v = np.zeros(n, np.int64)
    for j in range(k): v = v * A + c[j:j + n]
    return v

def _bag(lists, dim):
    rows = np.concatenate([np.full(len(v), i) for i, v in enumerate(lists)])
    cols = np.concatenate(lists)
    X = sp.csr_matrix((np.ones(len(cols), np.float32), (rows, cols)), shape=(len(lists), dim))
    X.sum_duplicates(); return drop_zero_cols(X)

def emb_kmer(seqs, alpha, k):  # Spike2Vec-style k-mer frequency vector
    A = len(alpha); return _bag([_int_kmers(codes(s, alpha), k, A) for s in seqs], A ** k)

def emb_minimizer(seqs, alpha, k, m):
    A = len(alpha); out = []
    for s in seqs:
        c = codes(s, alpha); f = _int_kmers(c, m, A); r = _int_kmers(c[::-1], m, A)[::-1]
        can = np.minimum(f, r); w = k - m + 1
        if len(can) < w: out.append(can[:1]); continue
        mn = minimum_filter1d(can, w, origin=-(w // 2))[:len(can) - w + 1]
        keep = np.r_[True, mn[1:] != mn[:-1]]; out.append(mn[keep])
    return _bag(out, A ** m)

def emb_spaced(seqs, alpha, g, k):  # gapped k-mers: k fixed positions spread over a g-window
    A = len(alpha); pos = np.round(np.linspace(0, g - 1, k)).astype(int); out = []
    for s in seqs:
        c = codes(s, alpha); n = len(c) - g + 1
        if n <= 0: out.append(np.zeros(0, np.int64)); continue
        v = np.zeros(n, np.int64)
        for p in pos: v = v * A + c[p:p + n]
        out.append(v)
    return _bag(out, A ** k)

def embeddings(seqs, alpha):
    nt = is_nt(alpha)
    return {'OHE': lambda: emb_ohe(seqs, alpha),
            'Spike2Vec': lambda: emb_kmer(seqs, alpha, 6 if nt else 3),
            'Minimizer': lambda: emb_minimizer(seqs, alpha, 12 if nt else 9, 6 if nt else 3),
            'SpacedKmer': lambda: emb_spaced(seqs, alpha, 9, 6 if nt else 4)}

# ---------------- kernels ----------------
@njit(parallel=True, fastmath=True)
def _common(indptr, indices, data, M1, M2):
    n = len(indptr) - 1
    for i in prange(n):
        for j in range(i, n):
            a, ae, b, be = indptr[i], indptr[i + 1], indptr[j], indptr[j + 1]
            s1 = 0.0; s2 = 0.0
            while a < ae and b < be:
                ia = indices[a]; ib = indices[b]
                if ia == ib:
                    x = data[a]; y = data[b]
                    s1 += min(x, y); s2 += 2.0 * x * y / (x + y); a += 1; b += 1
                elif ia < ib: a += 1
                else: b += 1
            M1[i, j] = s1; M1[j, i] = s1; M2[i, j] = s2; M2[j, i] = s2

def common_terms(X):
    """M1 = sum_i min(x,y); M2 = sum_i 2xy/(x+y) over shared support (exact)."""
    X = X.tocsr(); X.sort_indices()
    if np.all(X.data == 1):  # binary: min(x,y)=xy, 2xy/(x+y)=xy
        G = gram(X); return G, G
    n = X.shape[0]; M1 = np.zeros((n, n)); M2 = np.zeros((n, n))
    _common(X.indptr.astype(np.int64), X.indices.astype(np.int64), X.data.astype(np.float64), M1, M2)
    return M1, M2

def gram(X):
    if sp.issparse(X) and X.nnz > 2e7:  # very wide one-hot matrices: dense multithreaded BLAS is much faster
        Xd = X.toarray().astype(np.float32); return (Xd @ Xd.T).astype(np.float64)
    G = X @ X.T
    return np.asarray(G.todense() if sp.issparse(G) else G, dtype=np.float64)

def _med(M):
    iu = np.triu_indices(M.shape[0], 1)
    v = M[iu]; v = v[v > 0] if np.any(v > 0) else v
    rng = np.random.default_rng(0)
    if len(v) > 2_000_000: v = rng.choice(v, 2_000_000, replace=False)
    return float(np.median(v)) if len(v) else 1.0

def sqdist_from_gram(G):
    d = np.diag(G); return np.maximum(d[:, None] + d[None, :] - 2 * G, 0)

def isolation_kernel(X, psi=32, t=100, seed=0):
    rng = np.random.default_rng(seed); n = X.shape[0]
    X = X.tocsr().astype(np.float64); sq = np.asarray(X.multiply(X).sum(1)).ravel(); cols = []
    for r in range(t):
        idx = rng.choice(n, psi, replace=False); C = X[idx]
        d = sq[:, None] - 2 * np.asarray((X @ C.T).todense()) + sq[idx][None, :]
        cols.append(np.argmin(d, 1) + r * psi)
    P = sp.csr_matrix((np.ones(n * t), (np.repeat(np.arange(n), t), np.stack(cols, 1).ravel())), shape=(n, psi * t))
    return gram(P) / t

KERNELS = ['Cosine', 'Linear', 'Polynomial', 'Gaussian', 'Isolation', 'Laplacian', 'Sigmoid', 'ChiSquared', 'AdditiveChi2']

def kernel(name, X, psi=32):
    """returns (K, seconds). Bandwidths/scales set by median heuristic (documented)."""
    t0 = time.time()
    if name == 'Isolation':
        K = isolation_kernel(X, psi)
    elif name in ('Laplacian', 'ChiSquared', 'AdditiveChi2'):
        M1, M2 = common_terms(X); s = np.asarray(X.sum(1), np.float64).ravel()
        if name == 'Laplacian':
            L1 = np.maximum(s[:, None] + s[None, :] - 2 * M1, 0); K = np.exp(-L1 / _med(L1))
        elif name == 'ChiSquared':
            C = np.maximum(s[:, None] + s[None, :] - 2 * M2, 0); K = np.exp(-C / _med(C))
        else:
            K = M2
    else:
        G = gram(X)
        if name == 'Linear': K = G
        elif name == 'Cosine':
            d = np.sqrt(np.maximum(np.diag(G), 1e-12)); K = G / d[:, None] / d[None, :]
        elif name == 'Polynomial': K = (G / _med(G) + 1.0) ** 3
        elif name == 'Sigmoid': K = np.tanh(G / _med(G))
        elif name == 'Gaussian':
            D2 = sqdist_from_gram(G); K = np.exp(-D2 / _med(D2))
    return K, time.time() - t0

def kernel_dist(K):  # kernel-induced (feature-space) distance
    return np.sqrt(sqdist_from_gram(K))

# ---------------- t-SNE / UMAP ----------------
def tsne_from_dist(D, seed, perplexity=30, n_jobs=8):
    from openTSNE import TSNE, affinity, initialization
    D = np.ascontiguousarray(D, dtype=np.float64)
    aff = affinity.PerplexityBasedNN(D, perplexity=perplexity, metric='precomputed', n_jobs=n_jobs, random_state=seed)
    init = initialization.random(D.shape[0], random_state=seed)
    return np.asarray(TSNE(n_jobs=n_jobs, random_state=seed).fit(affinities=aff, initialization=init))

def umap_embed(X, seed, metric):
    import umap
    return umap.UMAP(metric=metric, random_state=seed, n_jobs=1).fit_transform(X)

# ---------------- neighbourhood preservation ----------------
def knn_from_dist(D, k=99, seed=0):
    D = D.copy(); np.fill_diagonal(D, np.inf)
    rng = np.random.default_rng(seed); D += rng.random(D.shape) * 1e-9 * (np.abs(D[np.isfinite(D)]).max() + 1)  # random tie-break
    idx = np.argpartition(D, k, axis=1)[:, :k]
    o = np.argsort(np.take_along_axis(D, idx, 1), 1); return np.take_along_axis(idx, o, 1)

def knn_ld(Y, k=99):
    from sklearn.neighbors import NearestNeighbors
    return NearestNeighbors(n_neighbors=k + 1).fit(Y).kneighbors(Y, return_distance=False)[:, 1:]

def auc_rnx(hd, ld):
    n, K = hd.shape; pos = np.full((n, K), 10 ** 6)
    for i in range(n):
        m = {v: p for p, v in enumerate(ld[i])}
        pos[i] = [m.get(v, 10 ** 6) for v in hd[i]]
    R = []
    for k in range(1, K + 1):
        Q = (pos[:, :k] < k).sum() / (n * k); R.append(((n - 1) * Q - k) / (n - 1 - k))
    R = np.array(R); w = 1 / np.arange(1, K + 1)
    return float((R * w).sum() / w.sum()), R

# ---------------- downstream ----------------
def classify(Y, y, seed=0):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.svm import SVC; from sklearn.naive_bayes import GaussianNB
    from sklearn.neural_network import MLPClassifier; from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier; from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier; from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    y = np.asarray(y); u, c = np.unique(y, return_counts=True); ok = np.isin(y, u[c >= 5])
    Y = StandardScaler().fit_transform(Y[ok]); y = y[ok]
    clfs = {'SVM': lambda: SVC(), 'NB': GaussianNB, 'MLP': lambda: MLPClassifier(max_iter=500, random_state=seed),
            'KNN': lambda: KNeighborsClassifier(5), 'RF': lambda: RandomForestClassifier(100, random_state=seed, n_jobs=4),
            'LR': lambda: LogisticRegression(max_iter=2000), 'DT': lambda: DecisionTreeClassifier(random_state=seed)}
    res = {}
    for name, f in clfs.items():
        acc, f1w, f1m, auc = [], [], [], []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Y, y):
            m = f().fit(Y[tr], y[tr]); p = m.predict(Y[te])
            acc.append(accuracy_score(y[te], p)); f1w.append(f1_score(y[te], p, average='weighted')); f1m.append(f1_score(y[te], p, average='macro'))
            if hasattr(m, 'predict_proba') and len(np.unique(y[te])) == len(m.classes_):
                try:
                    pr = m.predict_proba(Y[te])
                    auc.append(roc_auc_score(y[te], pr[:, 1] if pr.shape[1] == 2 else pr, multi_class='ovr', average='weighted', labels=m.classes_))
                except Exception: pass
        res[name] = dict(acc=np.mean(acc), f1w=np.mean(f1w), f1m=np.mean(f1m), auc=float(np.mean(auc)) if auc else None)
    return res

def cluster(Y, y, seed=0):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score
    kk = len(np.unique(y)); t0 = time.time(); lab = KMeans(kk, n_init=10, random_state=seed).fit_predict(Y); t = time.time() - t0
    ss = min(len(Y), 10000)
    return dict(k=kk, silhouette=float(silhouette_score(Y, lab, sample_size=ss, random_state=seed)), ch=float(calinski_harabasz_score(Y, lab)),
                db=float(davies_bouldin_score(Y, lab)), ari=float(adjusted_rand_score(y, lab)), nmi=float(normalized_mutual_info_score(y, lab)), sec=t)
