"""Figures: (1) t-SNE scatter grids, (2) R(k) curves, (3) runtime scaling, (4) mean-rank bars."""
import os, sys, glob, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import lib
R = lib.ROOT; F = f'{R}/figures'; os.makedirs(F, exist_ok=True)
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'figure.dpi': 200})
ORDER = ['Cosine', 'Linear', 'Polynomial', 'Gaussian', 'Isolation', 'Laplacian', 'Sigmoid', 'ChiSquared', 'AdditiveChi2', 'tSNE-Euclid', 'UMAP-Euclid', 'UMAP-Cosine']

def scatter_grid(ds, emb, seed=0):
    d = np.load(f'{R}/data/{ds}.npz', allow_pickle=True); y = np.asarray(d['labels']).astype(str)
    u, c = np.unique(y, return_counts=True); top = u[np.argsort(-c)][:10]
    yy = np.where(np.isin(y, top), y, 'Other'); cats = list(top) + (['Other'] if (yy == 'Other').any() else [])
    cmap = plt.get_cmap('tab10'); col = {k: (cmap(i) if k != 'Other' else (0.8, 0.8, 0.8, 1)) for i, k in enumerate(cats)}
    ms = [m for m in ORDER if os.path.exists(f'{R}/results/{ds}/{emb}/{m}_s{seed}_Y.npy')]
    if not ms: return
    nc = 4; nr = int(np.ceil(len(ms) / nc)); fig, ax = plt.subplots(nr, nc, figsize=(nc * 2.2, nr * 2.2)); ax = np.atleast_1d(ax).ravel()
    for a, m in zip(ax, ms):
        Y = np.load(f'{R}/results/{ds}/{emb}/{m}_s{seed}_Y.npy'); r = json.load(open(f'{R}/results/{ds}/{emb}/{m}_s{seed}.json'))
        for k in cats[::-1]:
            s = yy == k; a.scatter(Y[s, 0], Y[s, 1], s=1, c=[col[k]], linewidths=0, rasterized=True)
        a.set_title(f"{m}\nAUC={r['auc_euclid']:.3f}"); a.set_xticks([]); a.set_yticks([])
    for a in ax[len(ms):]: a.axis('off')
    h = [plt.Line2D([], [], marker='o', ls='', ms=4, color=col[k]) for k in cats]
    fig.legend(h, cats, loc='lower center', ncol=min(6, len(cats)), fontsize=6, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(f'{F}/scatter_{ds}_{emb}.pdf'); plt.close(fig)

def rnx_curves(ds):
    embs = ['OHE', 'Spike2Vec', 'Minimizer', 'SpacedKmer']; fig, ax = plt.subplots(1, 4, figsize=(10, 2.4), sharey=True)
    for a, e in zip(ax, embs):
        for m in ORDER[:10]:
            fs = glob.glob(f'{R}/results/{ds}/{e}/{m}_s[0-9].json')
            if not fs: continue
            Rk = np.mean([json.load(open(f))['R_euclid'] for f in fs], 0); a.plot(np.arange(1, len(Rk) + 1), Rk, label=m, lw=1)
        a.set_xscale('log'); a.set_title(f'{ds} / {e}'); a.set_xlabel('k')
    ax[0].set_ylabel('$R_{NX}(k)$'); ax[-1].legend(fontsize=5, ncol=2); fig.tight_layout(); fig.savefig(f'{F}/rnx_{ds}.pdf'); plt.close(fig)

def scaling():
    fs = glob.glob(f'{R}/results/scaling_*.jsonl')
    if not fs: return
    df = pd.concat([pd.read_json(f, lines=True) for f in fs]); fig, ax = plt.subplots(1, 2, figsize=(7, 2.6))
    for k, g in df.groupby('kernel'):
        m = g.groupby('n')['kernel_sec'].mean(); ax[0].plot(m.index, m.values, marker='o', ms=3, lw=1, label=k)
    ax[0].set_xscale('log'); ax[0].set_yscale('log'); ax[0].set_xlabel('number of sequences'); ax[0].set_ylabel('kernel time (s)'); ax[0].legend(fontsize=5, ncol=2)
    t = df.dropna(subset=['tsne_sec']).groupby('n')['tsne_sec'].mean(); ax[1].plot(t.index, t.values, marker='o', ms=3)
    ax[1].set_xscale('log'); ax[1].set_yscale('log'); ax[1].set_xlabel('number of sequences'); ax[1].set_ylabel('t-SNE time (s)')
    fig.tight_layout(); fig.savefig(f'{F}/scaling.pdf'); plt.close(fig)
    sl = {k: np.polyfit(np.log(g.groupby('n')['kernel_sec'].mean().index), np.log(g.groupby('n')['kernel_sec'].mean().values), 1)[0] for k, g in df.groupby('kernel')}
    pd.Series(sl, name='loglog_slope').to_csv(f'{R}/tables/scaling_slopes.csv'); print('slopes', {k: round(v, 2) for k, v in sl.items()})

if __name__ == '__main__':
    for f in sorted(glob.glob(f'{R}/data/*.npz')):
        ds = os.path.basename(f)[:-4]
        for e in ['OHE', 'Spike2Vec', 'Minimizer', 'SpacedKmer']:
            try: scatter_grid(ds, e)
            except Exception as ex: print('scatter fail', ds, e, ex)
        try: rnx_curves(ds)
        except Exception as ex: print('rnx fail', ds, ex)
    scaling(); print('PLOTS_DONE', sorted(os.listdir(F)))
