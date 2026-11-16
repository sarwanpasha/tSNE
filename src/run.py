import os, sys, argparse
ap = argparse.ArgumentParser(); ap.add_argument('ds'); ap.add_argument('emb')
ap.add_argument('--seeds', type=int, default=5); ap.add_argument('--threads', type=int, default=8)
ap.add_argument('--kernels', default='all'); ap.add_argument('--no-downstream', action='store_true')
a = ap.parse_args()
for v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS']: os.environ[v] = str(a.threads)
import json, time, numpy as np, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import lib

R = lib.ROOT; d = np.load(f'{R}/data/{a.ds}.npz', allow_pickle=True)
seqs, y, alpha = list(d['seqs']), np.asarray(d['labels']).astype(str), str(d['alpha'])
ef = f'{R}/data/emb/{a.ds}_{a.emb}.npz'; os.makedirs(os.path.dirname(ef), exist_ok=True)
if os.path.exists(ef): X = sp.load_npz(ef)
else:
    t0 = time.time(); X = lib.embeddings(seqs, alpha)[a.emb](); sp.save_npz(ef, X)
    print('embedding', X.shape, f'{time.time()-t0:.1f}s', flush=True)
out = f'{R}/results/{a.ds}/{a.emb}'; os.makedirs(out, exist_ok=True)
G = lib.gram(X); De = np.sqrt(lib.sqdist_from_gram(G)); del G
hdE = lib.knn_from_dist(De)

def done(m): return all(os.path.exists(f'{out}/{m}_s{s}.json') for s in range(a.seeds))

def evaluate(m, Ymaker, hdK, extra):
    for s in range(a.seeds):
        f = f'{out}/{m}_s{s}.json'
        if os.path.exists(f): continue
        t0 = time.time(); Y = Ymaker(s); tt = time.time() - t0
        aE, RE = lib.auc_rnx(hdE, lib.knn_ld(Y)); aK, RK = lib.auc_rnx(hdK, lib.knn_ld(Y))
        r = dict(ds=a.ds, emb=a.emb, method=m, seed=s, n=len(y), dim=X.shape[1], tsne_sec=tt,
                 auc_euclid=aE, auc_kernel=aK, R_euclid=RE.tolist(), R_kernel=RK.tolist(), **extra)
        if not a.no_downstream:
            r['clf'] = lib.classify(Y, y, s); r['clu'] = lib.cluster(Y, y, s)
        np.save(f'{out}/{m}_s{s}_Y.npy', Y.astype(np.float32))
        json.dump(r, open(f, 'w')); print(m, s, f'aucE={aE:.3f} aucK={aK:.3f} t={tt:.0f}s', flush=True)

ks = lib.KERNELS if a.kernels == 'all' else a.kernels.split(',')
for k in ks:
    if done(k): continue
    K, kt = lib.kernel(k, X); D = lib.kernel_dist(K); del K
    print(k, f'kernel {kt:.1f}s', flush=True)
    hdK = lib.knn_from_dist(D)
    evaluate(k, lambda s: lib.tsne_from_dist(D, s, n_jobs=a.threads), hdK, dict(kernel_sec=kt))
    del D
if a.kernels == 'all':
    if not done('tSNE-Euclid'):
        evaluate('tSNE-Euclid', lambda s: lib.tsne_from_dist(De, s, n_jobs=a.threads), hdE, {})
    Xn = sp.csr_matrix(lib.normalize(X)); Dc = np.sqrt(lib.sqdist_from_gram(lib.gram(Xn))); hdC = lib.knn_from_dist(Dc); del Dc
    if not done('UMAP-Euclid'): evaluate('UMAP-Euclid', lambda s: lib.umap_embed(X, s, 'euclidean'), hdE, {})
    if not done('UMAP-Cosine'): evaluate('UMAP-Cosine', lambda s: lib.umap_embed(X, s, 'cosine'), hdC, {})
print('ALLDONE', a.ds, a.emb, flush=True)
