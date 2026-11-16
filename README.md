# Kernel t-SNE and Neighbourhood-Consensus Multi-Kernel t-SNE for biological sequences

This repository contains the code for embedding biological sequences, computing kernel matrices, running kernel t-SNE, and the Neighbourhood-Consensus Multi-Kernel t-SNE (NC-MKt-SNE) method, together with the evaluation pipeline (neighbourhood preservation, classification, clustering, runtime scaling, and statistical tests).

## Contents

| File | Purpose |
|---|---|
| `src/lib.py` | Sequence embeddings (one-hot, Spike2Vec k-mers, minimizers, spaced k-mers), nine kernels (cosine, linear, polynomial, Gaussian, isolation, Laplacian, sigmoid, chi-squared, additive chi-squared), kernel-induced distances, t-SNE and UMAP wrappers, `AUC_RNX`, classification and clustering |
| `src/run.py` | Kernel t-SNE for every kernel plus Euclidean t-SNE and UMAP baselines for one dataset/embedding pair, five seeds |
| `src/consensus.py` | NC-MKt-SNE and its two ablations (uniform weights `MK-mean`, per-kernel weights `NC-global`) |
| `src/scaling.py` | Runtime of kernel computation and t-SNE for increasing numbers of sequences |
| `src/aggregate.py` | Collects all per-seed results, average ranks, Friedman/Nemenyi and Holm-corrected Wilcoxon tests |
| `src/plots.py`, `src/figs.py`, `src/grid.py` | Plots of embeddings, `R_NX(k)` curves and runtime |
| `src/prepare_public.py` | Builds the public benchmark datasets after downloading |
| `src/prepare_custom.py` | Converts your own labelled FASTA/CSV file into the input format |
| `scripts/fetch_public.sh` | Downloads the public datasets (UCI anticancer peptides, NCBI Dengue genomes, Influenza A hemagglutinin, SARS-CoV-2 spike proteins with Pango lineages) |
| `scripts/run_all.sh` | Runs the full pipeline for a list of datasets |

## Installation

```bash
conda create -n tsne python=3.11 -y && conda activate tsne
pip install -r requirements.txt
```

The public download script additionally needs the NCBI `datasets` command line tool.

## Usage

All outputs are written to the directory given by the environment variable `TSNE_ROOT` (default: `./workspace`).

```bash
export TSNE_ROOT=workspace
bash scripts/fetch_public.sh                 # download public data
python src/prepare_public.py                 # build workspace/data/*.npz
python src/run.py InfluenzaHA Spike2Vec      # all kernels + baselines, 5 seeds
python src/consensus.py InfluenzaHA Spike2Vec   # NC-MKt-SNE
python src/aggregate.py                      # tables/ with summaries and tests
```

To run everything for several datasets in parallel:

```bash
bash scripts/run_all.sh "LungCancer BreastCancer Dengue InfluenzaHA CoVNCBI" 8 8
```

Your own data can be used with `python src/prepare_custom.py NAME file.fasta` (label = last `|`-separated field of each header) or a CSV file with columns `sequence,label`.

## Method in brief

For each kernel `K_m`, the kernel-induced distance `d_m(x,y) = sqrt(K_m(x,x) + K_m(y,y) - 2 K_m(x,y))` is converted into perplexity-calibrated t-SNE affinities `p^(m)(j|i)`. NC-MKt-SNE scores, for each point `i` and kernel `m`, the Jaccard agreement `a_{m,i}` of its 10 nearest neighbours with those of the other kernels, sets point-specific weights `w_{m,i} = softmax_m(a_{m,i} / tau)`, and runs t-SNE on the fused affinities `p(j|i) = sum_m w_{m,i} p^(m)(j|i)`.

## Data availability

The SARS-CoV-2 datasets derived from GISAID cannot be redistributed under the GISAID terms of use. All other datasets are public and can be reproduced with `scripts/fetch_public.sh`.

## License

MIT
