"""Scalable NC-MKt-SNE: approximate kNN per kernel (pynndescent), no dense kernel matrices.
Kernels whose neighbor sets coincide with a metric (Lemma/Proposition in the paper):
  Gaussian (Euclidean kNN), cosine (cosine kNN), Laplacian (Manhattan kNN); kernel distances are
  then evaluated exactly on the candidate neighbors. Reports time and memory versus n.
usage: rev_scale.py FASTA_OR_DS [--sizes 10000,25000,50000,100000,150000] [--threads 32]
       rev_scale.py --validate DS EMB   (compare with exact NC-MKt-SNE on the same 3 kernels)"""
import argparse, os, time, resource
ap = argparse.ArgumentParser(); ap.add_argument('src'); ap.add_argument('emb', nargs='?', default='Spike2Vec')
ap.add_argument('--sizes', default='10000,25000,50000,100000,150000'); ap.add_argument('--threads', type=int, default=32)
ap.add_argument('--validate', action='store_true'); ap.add_argument('--seeds', type=int, default=3)
a = ap.parse_args()
for v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS']: os.environ[v] = str(a.threads)
import numpy as np, scipy.sparse as sp, json
from rev_common import *
K = 90; KR = 10; TAU = 0.05

def med_sample(f, X, rng, m=2000):
    i = rng.choice(X.shape[0], min(m, X.shape[0]), replace=False); D = f(X[i]); return float(np.median(D[np.triu_indices(len(i), 1)]))

def knn(X, metric, seed):
    from pynndescent import NNDescent
    idx, d = NNDescent(X, metric=metric, n_neighbors=K + 1, random_state=seed, n_jobs=a.threads, low_memory=True).neighbor_graph
    return idx[:, 1:], d[:, 1:]

def fast_nc(X, seed):
    from sklearn.metrics import pairwise_distances
    rng = np.random.default_rng(seed); T = {}; t0 = time.time()
    i, dd = knn(X, 'euclidean', seed); h2 = med_sample(lambda Z: pairwise_distances(Z) ** 2, X, rng)
    T['Gaussian'] = (i, np.sqrt(2 * (1 - np.exp(-dd ** 2 / h2))))
    i, dd = knn(X, 'cosine', seed); T['Cosine'] = (i, np.sqrt(np.maximum(2 * dd, 0)))
    i, dd = knn(X, 'manhattan', seed); h1 = med_sample(lambda Z: pairwise_distances(Z, metric='manhattan'), X, rng)
    T['Laplacian'] = (i, np.sqrt(2 * (1 - np.exp(-dd / h1))))
    tk = time.time() - t0
    IDX = np.stack([T[k][0] for k in T]); PP = np.stack([cond_from_knn(T[k][0], T[k][1], 30) for k in T])
    A = reliability(IDX, list(range(3)), KR); W = weights(A, TAU); J = joint(IDX, PP, list(range(3)), W)
    return J, tk, time.time() - t0 - tk

if a.validate:
    ds, emb = a.src, a.emb; X = X_of(ds, emb).astype(np.float32); y = labels(ds); hdE = hd_euclid(ds, emb)
    IDXc, PPc, _ = load_block(ds, emb); ks = [lib.KERNELS.index(k) for k in ['Gaussian', 'Cosine', 'Laplacian']]
    Jex = joint(IDXc, PPc, ks, weights(reliability(IDXc, ks, KR), TAU))
    for s in range(a.seeds):
        if not done(ds, emb, 'NC3-fast', s):
            J, tk, tc = fast_nc(X, s); Y, t = run_tsne(J, s, a.threads); evaluate_and_save(ds, emb, 'NC3-fast', s, Y, hdE, y, dict(knn_sec=tk, consensus_sec=tc, tsne_sec=t))
        if not done(ds, emb, 'NC3-exact', s):
            Y, t = run_tsne(Jex, s, a.threads); evaluate_and_save(ds, emb, 'NC3-exact', s, Y, hdE, y, dict(tsne_sec=t))
else:
    seqs = []; cur = []
    for line in open(a.src):
        if line.startswith('>'):
            if cur: seqs.append(''.join(cur)); cur = []
        else: cur.append(line.strip())
    if cur: seqs.append(''.join(cur))
    seqs = list(dict.fromkeys(seqs)); print('unique sequences', len(seqs), flush=True)
    out = f'{R}/tables/scale_fast.jsonl'
    for n in map(int, a.sizes.split(',')):
        if n > len(seqs): continue
        sub = [seqs[i] for i in np.random.default_rng(0).choice(len(seqs), n, replace=False)]
        t0 = time.time(); cl, al = lib.clean(sub); X = lib.drop_zero_cols(lib.emb_kmer(cl, al, 3)); te = time.time() - t0
        J, tk, tc = fast_nc(X, 0); Y, t = run_tsne(J, 0, a.threads)
        rec = dict(n=n, embed_sec=te, knn_sec=tk, consensus_sec=tc, tsne_sec=t, total_sec=te + tk + tc + t, maxrss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, threads=a.threads)
        open(out, 'a').write(json.dumps(rec) + '\n'); print(rec, flush=True)
print('SCALE_DONE', flush=True)
