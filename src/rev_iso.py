"""Sensitivity of the isolation kernel to the number of sampled points psi (t = 100 partitions).
usage: rev_iso.py DS EMB [--psis 8,16,64,128,256] [--seeds 3] [--threads 4]"""
import argparse, os
ap = argparse.ArgumentParser(); ap.add_argument('ds'); ap.add_argument('emb')
ap.add_argument('--psis', default='8,16,64,128,256'); ap.add_argument('--seeds', type=int, default=3)
ap.add_argument('--threads', type=int, default=4); a = ap.parse_args()
for v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS']: os.environ[v] = str(a.threads)
from rev_common import *
X = X_of(a.ds, a.emb); y = labels(a.ds); hdE = hd_euclid(a.ds, a.emb)
for psi in map(int, a.psis.split(',')):
    name = f'Iso-psi{psi}'
    if all(done(a.ds, a.emb, name, s) for s in range(a.seeds)): continue
    psi_eff = min(psi, X.shape[0] - 1)
    K, tk = lib.kernel('Isolation', X, psi_eff); D = lib.kernel_dist(K); del K
    for s in range(a.seeds):
        if done(a.ds, a.emb, name, s): continue
        t0 = time.time(); Y = lib.tsne_from_dist(D, s, n_jobs=a.threads)
        evaluate_and_save(a.ds, a.emb, name, s, Y, hdE, y, dict(psi=psi_eff, kernel_sec=tk, tsne_sec=time.time() - t0))
    del D
print('ISO_DONE', a.ds, a.emb, flush=True)
