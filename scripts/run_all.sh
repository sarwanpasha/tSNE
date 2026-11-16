#!/bin/bash
# Usage: bash scripts/run_all.sh "DATASET1 DATASET2 ..." N_PARALLEL THREADS_PER_JOB
# Runs all kernels, baselines and NC-MKt-SNE for every dataset/embedding pair (resumable).
PY=${PYTHON:-python}; mkdir -p ${TSNE_ROOT:-workspace}/logs
for ds in $1; do for e in Spike2Vec Minimizer SpacedKmer OHE; do echo "$ds $e"; done; done |
  xargs -P ${2:-4} -L 1 sh -c "$PY src/run.py \$0 \$1 --threads ${3:-8} && $PY src/consensus.py \$0 \$1 --threads ${3:-8}"
$PY src/aggregate.py
