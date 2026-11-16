"""Build workspace/data/{name}.npz (seqs, labels, alpha) for the public benchmark datasets."""
import os, sys, glob, csv, re, numpy as np
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import lib
R = lib.ROOT; RAW = f'{R}/data/raw'; os.makedirs(f'{R}/data', exist_ok=True)

def fasta(path):
    out, h, buf = [], None, []
    for line in open(path):
        line = line.rstrip()
        if line.startswith('>'):
            if h is not None: out.append((h, ''.join(buf)))
            h, buf = line[1:], []
        else: buf.append(line)
    if h is not None: out.append((h, ''.join(buf)))
    return out

def save(name, seqs, labels, minlen=1, minclass=1):
    keep = [i for i, s in enumerate(seqs) if len(s) >= minlen and labels[i]]
    cc = Counter(labels[i] for i in keep); keep = [i for i in keep if cc[labels[i]] >= minclass]
    seqs = [seqs[i] for i in keep]; labels = [labels[i] for i in keep]
    seqs, alpha = lib.clean(seqs)
    np.savez(f'{R}/data/{name}.npz', seqs=np.array(seqs, dtype=object), labels=np.array(labels), alpha=alpha)
    print(name, len(seqs), 'sequences,', len(set(labels)), 'classes')

def build(name):
    if name in ('BreastCancer', 'LungCancer'):
        rows = list(csv.reader(open(f'{RAW}/ACPs_{"Breast" if name == "BreastCancer" else "Lung"}_cancer.csv')))
        save(name, [r[1] for r in rows[1:]], [r[2] for r in rows[1:]])
    elif name == 'Dengue':
        S, L = [], []
        for st in ['DENV1', 'DENV2', 'DENV3', 'DENV4']:
            for h, s in fasta(f'{RAW}/dengue_{st}/ncbi_dataset/data/genomic.fna'): S.append(s); L.append(st)
        save(name, S, L, minlen=9000)
    elif name == 'InfluenzaHA':
        S, L = [], []
        for st in ['H1N1', 'H3N2', 'H5N1', 'H7N9', 'H9N2']:
            for h, s in fasta(f'{RAW}/flu_{st}.faa'): S.append(s); L.append(st)
        save(name, S, L, minclass=50)
    elif name == 'CoVNCBI':
        S, L = [], []
        for d in sorted(glob.glob(f'{RAW}/cov_*')):
            if not os.path.isdir(d): continue
            lin = os.path.basename(d)[4:]
            for h, s in fasta(f'{d}/ncbi_dataset/data/protein.faa'):
                if 'surface glycoprotein' in h and 1200 <= len(s) <= 1300: S.append(s); L.append(lin)
        save(name, S, L, minclass=50)

if __name__ == '__main__':
    for n in (sys.argv[1:] or ['BreastCancer', 'LungCancer', 'Dengue', 'InfluenzaHA', 'CoVNCBI']): build(n)
