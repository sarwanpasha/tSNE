#!/bin/bash
# Second set of workers for the revision sweeps that walks the blocks in reverse order (smallest datasets first),
# so that the two worker sets meet in the middle. Finished seeds are skipped. usage: run_reverse.sh PYTHON CODE_DIR
P=${1:-python}; C=${2:-src}; cd "$(dirname "$C")" || exit 1; C=$(basename "$C"); mkdir -p logs
BLOCKS=""; for d in LungCancer BreastCancer Dengue Host Spike7k InfluenzaHA Genome ShortRead CoVNCBI Rabies; do
  for e in OHE SpacedKmer Minimizer Spike2Vec; do BLOCKS="$BLOCKS $d:$e"; done; done
launch() { # name jobs threads script mode
  tmux new-session -d -s tsne_rev_$1_r "echo $BLOCKS | tr ' ' '\n' | grep . | xargs -P $2 -I{} sh -c 'b={}; $P $C/$4 $5 \${b%%:*} \${b##*:} --threads $3' > logs/rev_$1_r.log 2>&1; echo ALL_FINISHED >> logs/rev_$1_r.log"
}
launch dedup 3 4 rev_consensus.py dedup
launch tau 3 4 rev_consensus.py tau
launch kr 3 4 rev_consensus.py kr
launch pca 3 4 rev_consensus.py pca
launch msc 3 4 rev_baselines.py multiscale
launch meta 2 6 rev_baselines.py metavis
tmux ls | grep _r:
