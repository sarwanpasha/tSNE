"""Figures of embeddings, R_NX(k) curves and runtime (no titles, large fonts). Output: figures/main/*.pdf"""
import os, sys, glob, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import lib
R = lib.ROOT; F = f'{R}/figures/main'; os.makedirs(F, exist_ok=True)
plt.rcParams.update({'font.size': 16, 'axes.labelsize': 18, 'xtick.labelsize': 15, 'ytick.labelsize': 15, 'legend.fontsize': 13, 'pdf.fonttype': 42})

def scatter(ds, emb, methods, seed=0, ntop=8):
    d = np.load(f'{R}/data/{ds}.npz', allow_pickle=True); y = np.asarray(d['labels']).astype(str)
    u, c = np.unique(y, return_counts=True); top = list(u[np.argsort(-c)][:ntop])
    yy = np.where(np.isin(y, top), y, 'Other'); cats = top + (['Other'] if (yy == 'Other').any() else [])
    cm = plt.get_cmap('tab10'); col = {k: (cm(i) if k != 'Other' else (0.75, 0.75, 0.75, 1)) for i, k in enumerate(cats)}
    for m in methods:
        Y = np.load(f'{R}/results/{ds}/{emb}/{m}_s{seed}_Y.npy')
        fig, ax = plt.subplots(figsize=(4, 4))
        for k in cats[::-1]:
            s = yy == k; ax.scatter(Y[s, 0], Y[s, 1], s=3, c=[col[k]], linewidths=0, rasterized=True)
        ax.set_xticks([]); ax.set_yticks([]); fig.tight_layout(pad=0.1); fig.savefig(f'{F}/scatter_{ds}_{emb}_{m}.pdf', dpi=300); plt.close(fig)
    fig = plt.figure(figsize=(12, 0.6)); h = [plt.Line2D([], [], marker='o', ls='', ms=10, color=col[k]) for k in cats]
    fig.legend(h, cats, loc='center', ncol=len(cats), frameon=False, fontsize=15, handletextpad=0.2, columnspacing=1.0)
    fig.savefig(f'{F}/scatter_{ds}_legend.pdf', bbox_inches='tight'); plt.close(fig)

def rnx(ds, emb, methods, names):
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    for m, nm in zip(methods, names):
        fs = glob.glob(f'{R}/results/{ds}/{emb}/{m}_s[0-9].json')
        Rk = np.mean([json.load(open(f))['R_euclid'] for f in fs], 0)
        ax.plot(np.arange(1, len(Rk) + 1), Rk, lw=2.2 if m.startswith('NC') else 1.6, label=nm)
    ax.set_xscale('log'); ax.set_xlabel('$k$'); ax.set_ylabel('$R_{NX}(k)$'); ax.legend(frameon=False, ncol=1)
    fig.tight_layout(); fig.savefig(f'{F}/rnx_{ds}_{emb}.pdf'); plt.close(fig)

def scaling(ds):
    df = pd.concat([pd.read_json(f, lines=True) for f in glob.glob(f'{R}/results/scaling_{ds}_*.jsonl')])
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    names = {'ChiSquared': 'Chi-squared', 'AdditiveChi2': 'Add. chi2'}
    for k, g in df.groupby('kernel'):
        m = g.groupby('n')['kernel_sec'].mean(); ax.plot(m.index, m.values, marker='o', ms=4, lw=1.6, label=names.get(k, k))
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('Number of sequences $n$'); ax.set_ylabel('Kernel time (s)')
    ax.legend(frameon=False, ncol=2, fontsize=11); fig.tight_layout(); fig.savefig(f'{F}/scaling_kernel_{ds}.pdf'); plt.close(fig)

M = ['NC-MKtSNE', 'Gaussian', 'Cosine', 'Isolation', 'Sigmoid', 'UMAP-Euclid']
scatter('Spike7k', 'Spike2Vec', M); scatter('InfluenzaHA', 'Spike2Vec', M)
rnx('Rabies', 'Spike2Vec', ['NC-MKtSNE', 'Gaussian', 'Cosine', 'Laplacian', 'Isolation', 'Sigmoid', 'UMAP-Euclid'],
    ['NC-MKt-SNE', 'Gaussian', 'Cosine', 'Laplacian', 'Isolation', 'Sigmoid', 'UMAP'])
rnx('Genome', 'Minimizer', ['NC-MKtSNE', 'Gaussian', 'Cosine', 'Laplacian', 'Isolation', 'Sigmoid', 'UMAP-Euclid'],
    ['NC-MKt-SNE', 'Gaussian', 'Cosine', 'Laplacian', 'Isolation', 'Sigmoid', 'UMAP'])
scaling('Rabies')
print('FIGS_DONE', sorted(os.listdir(F)))
