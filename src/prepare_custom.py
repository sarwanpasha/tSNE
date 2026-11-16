"""Convert your own labelled sequences (FASTA with label as the last '|' field, or a CSV with
columns 'sequence,label') into workspace/data/{name}.npz.
Usage: python src/prepare_custom.py NAME path/to/file.fasta|file.csv"""
import os, sys, csv, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import lib
name, path = sys.argv[1], sys.argv[2]; S, L = [], []
if path.endswith('.csv'):
    for r in csv.DictReader(open(path)): S.append(r['sequence']); L.append(r['label'])
else:
    h, buf = None, []
    for line in list(open(path)) + ['>']:
        line = line.strip()
        if line.startswith('>'):
            if h is not None: S.append(''.join(buf)); L.append(h.split('|')[-1])
            h, buf = line[1:], []
        else: buf.append(line)
S, alpha = lib.clean(S); os.makedirs(f'{lib.ROOT}/data', exist_ok=True)
np.savez(f'{lib.ROOT}/data/{name}.npz', seqs=np.array(S, dtype=object), labels=np.array(L), alpha=alpha)
print(name, len(S), 'sequences')
