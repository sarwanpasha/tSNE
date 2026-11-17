# Kernel t-SNE and Neighbourhood-Consensus Multi-Kernel t-SNE for biological sequences

This repository contains the code for embedding biological sequences, computing kernel matrices, running kernel t-SNE, and the Neighbourhood-Consensus Multi-Kernel t-SNE (NC-MKt-SNE) method, together with the evaluation pipeline (neighbourhood preservation, classification, clustering, runtime scaling, and statistical tests), additional baselines, sensitivity analyses, kernel-independent evaluation, and a scalable approximate version of the consensus.

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
| `src/rev_common.py` | Shared helpers for the additional experiments: loading the cached per-kernel neighbour graphs, reliability scores, weights, fused affinities, t-SNE with random or PCA initialisation, and evaluation |
| `src/rev_consensus.py` | Consensus variants on the cached graphs: de-duplicated kernel set (`dedup`), temperature `tau` and reliability neighbourhood `k_r` sweeps (`tau`, `kr`), PCA initialisation (`pca`), and m-SNE with alternating per-kernel weights (`msne`) |
| `src/rev_baselines.py` | Multiscale t-SNE over perplexities 4 to 64 (`multiscale`) and the spectral meta-visualisation of Ma et al. (`metavis`) |
| `src/rev_iso.py` | Sensitivity of the isolation kernel to the number of sampled points `psi` |
| `src/rev_indep.py` | `AUC_RNX` against kernel-independent references: normalised Levenshtein distance for amino acid sequences and MinHash (Mash) distance for nucleotide sequences, on a fixed random subset |
| `src/rev_scale.py` | Scalable consensus of the Gaussian, cosine and Laplacian kernels with approximate nearest neighbours (NN-descent), runtime and memory for growing `n`, and comparison with the exact consensus (`--validate`) |
| `src/rev_case.py` | Case studies: distance between parent and child Pango lineages on the maps, and consensus reliability versus ambiguous characters |
| `src/rev_stats.py` | Results with and without one-hot encoding, dataset-level ranks and Friedman tests, paired effect sizes with bootstrap intervals, and summaries of all additional experiments |
| `src/rev_report.py` | Compact text report of the additional experiments (sections `methods`, `effects`, `scale`, `case`) |
| `scripts/fetch_scale.sh` | Downloads full-length SARS-CoV-2 spike proteins from NCBI Protein for the scalability experiment |
| `scripts/run_revision.sh` | Launches all additional experiments in parallel tmux sessions |
| `scripts/run_reverse.sh` | Starts a second set of workers that processes the dataset/embedding blocks in reverse order, so the two sets meet in the middle |

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

## Additional experiments

These scripts reuse the per-kernel neighbour graphs cached by `src/consensus.py`, so most of them do not recompute any kernel matrix. Run them from the directory that contains `src/` after the main pipeline has finished.

```bash
python src/rev_consensus.py tau Rabies Spike2Vec --threads 4     # one sweep for one block
python src/rev_baselines.py multiscale Rabies Spike2Vec --threads 4
python src/rev_iso.py Rabies Spike2Vec --threads 4
python src/rev_indep.py Rabies --threads 8                       # kernel-independent references
bash scripts/fetch_scale.sh workspace/data/raw 400000
python src/rev_scale.py workspace/data/raw/spike_scale.fasta --threads 16
python src/rev_scale.py Rabies Spike2Vec --validate --threads 8  # approximate versus exact consensus
python src/rev_case.py
python src/rev_stats.py && python src/rev_report.py all
```

All blocks can be launched at once in parallel tmux sessions with `bash scripts/run_revision.sh python src`, optionally followed by `bash scripts/run_reverse.sh python src` to add a second set of workers. Finished seeds are skipped, so interrupted runs can simply be restarted.

Your own data can be used with `python src/prepare_custom.py NAME file.fasta` (label = last `|`-separated field of each header) or a CSV file with columns `sequence,label`.

## Method in brief

For each kernel `K_m`, the kernel-induced distance `d_m(x,y) = sqrt(K_m(x,x) + K_m(y,y) - 2 K_m(x,y))` is converted into perplexity-calibrated t-SNE affinities `p^(m)(j|i)`. NC-MKt-SNE scores, for each point `i` and kernel `m`, the Jaccard agreement `a_{m,i}` of its 10 nearest neighbours with those of the other kernels, sets point-specific weights `w_{m,i} = softmax_m(a_{m,i} / tau)`, and runs t-SNE on the fused affinities `p(j|i) = sum_m w_{m,i} p^(m)(j|i)`. With uniform weights (`MK-mean`), the fused objective has the same minimisers as multi-SNE with uniform weights. For large collections, the Gaussian, cosine and Laplacian kernels have the same nearest neighbours as the Euclidean, cosine and L1 distances, so `src/rev_scale.py` finds them with approximate nearest-neighbour search and never forms a kernel matrix.

## Data availability

The SARS-CoV-2 datasets derived from GISAID cannot be redistributed under the GISAID terms of use. All other datasets are public and can be reproduced with `scripts/fetch_public.sh`.

## License

MIT
