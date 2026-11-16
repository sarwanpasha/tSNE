"""Runtime scaling of kernel computation and t-SNE vs number of sequences (fixed threads)."""
import os, sys, argparse
ap = argparse.ArgumentParser(); ap.add_argument('ds'); ap.add_argument('emb'); ap.add_argument('--threads', type=int, default=16)
ap.add_argument('--sizes', default='1000,2000,4000,8000,16000'); ap.add_argument('--reps', type=int, default=3)
a = ap.parse_args()
for v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS']: os.environ[v] = str(a.threads)
import json, time, numpy as np, scipy.sparse as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import lib
X = sp.load_npz(f'{lib.ROOT}/data/emb/{a.ds}_{a.emb}.npz')
lib.kernel('ChiSquared', X[:50])  # numba warm-up (compile time excluded)
out = f'{lib.ROOT}/results/scaling_{a.ds}_{a.emb}.jsonl'
for n in map(int, a.sizes.split(',')):
    if n > X.shape[0]: continue
    for r in range(a.reps):
        idx = np.random.default_rng(r).choice(X.shape[0], n, replace=False); Xs = X[idx]
        for k in lib.KERNELS:
            K, kt = lib.kernel(k, Xs)
            rec = dict(ds=a.ds, emb=a.emb, n=n, rep=r, kernel=k, kernel_sec=kt, threads=a.threads)
            if k == 'Cosine' or r == 0:
                D = lib.kernel_dist(K); t0 = time.time(); lib.tsne_from_dist(D, r, n_jobs=a.threads); rec['tsne_sec'] = time.time() - t0
            open(out, 'a').write(json.dumps(rec) + '\n'); print(rec, flush=True)
print('SCALING_DONE')
