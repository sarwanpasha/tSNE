"""One 2x3 scatter grid (no titles) + bottom legend per dataset."""
import os, sys, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import lib
R = lib.ROOT; F = f'{R}/figures/main'; os.makedirs(F, exist_ok=True)
plt.rcParams.update({'pdf.fonttype': 42})
M = ['NC-MKtSNE', 'Gaussian', 'Cosine', 'Isolation', 'Sigmoid', 'UMAP-Euclid']
for ds, emb in [('Spike7k', 'Spike2Vec'), ('InfluenzaHA', 'Spike2Vec')]:
    d = np.load(f'{R}/data/{ds}.npz', allow_pickle=True); y = np.asarray(d['labels']).astype(str)
    u, c = np.unique(y, return_counts=True); top = list(u[np.argsort(-c)][:8])
    yy = np.where(np.isin(y, top), y, 'Other'); cats = top + (['Other'] if (yy == 'Other').any() else [])
    cm = plt.get_cmap('tab10'); col = {k: (cm(i) if k != 'Other' else (0.75, 0.75, 0.75, 1)) for i, k in enumerate(cats)}
    fig, axs = plt.subplots(2, 3, figsize=(12, 8.6))
    for ax, m in zip(axs.ravel(), M):
        Y = np.load(f'{R}/results/{ds}/{emb}/{m}_s0_Y.npy')
        for k in cats[::-1]:
            s = yy == k; ax.scatter(Y[s, 0], Y[s, 1], s=3, c=[col[k]], linewidths=0, rasterized=True)
        ax.set_xticks([]); ax.set_yticks([])
    h = [plt.Line2D([], [], marker='o', ls='', ms=11, color=col[k]) for k in cats]
    fig.legend(h, cats, loc='lower center', ncol=min(5, len(cats)), frameon=False, fontsize=17, handletextpad=0.2, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0.11, 1, 1), w_pad=0.4, h_pad=0.4); fig.savefig(f'{F}/grid_{ds}.pdf', dpi=300); plt.close(fig)
print('GRID_DONE')
