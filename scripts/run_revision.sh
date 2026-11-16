#!/bin/bash
# Launch all revision experiments in parallel tmux sessions. usage: run_revision.sh PYTHON CODE_DIR
P=${1:-python}; C=${2:-src}; cd "$(dirname "$C")" || exit 1; C=$(basename "$C"); mkdir -p logs
BLOCKS=""; for d in Rabies CoVNCBI ShortRead Genome InfluenzaHA Spike7k Host Dengue BreastCancer LungCancer; do
  for e in Spike2Vec Minimizer SpacedKmer OHE; do BLOCKS="$BLOCKS $d:$e"; done; done
launch() { # name jobs threads script mode
  tmux new-session -d -s tsne_rev_$1 "echo $BLOCKS | tr ' ' '\n' | grep . | xargs -P $2 -I{} sh -c 'b={}; $P $C/$4 $5 \${b%%:*} \${b##*:} --threads $3' > logs/rev_$1.log 2>&1; echo ALL_FINISHED >> logs/rev_$1.log"
}
launch dedup 3 4 rev_consensus.py dedup
launch tau 3 4 rev_consensus.py tau
launch kr 3 4 rev_consensus.py kr
launch pca 3 4 rev_consensus.py pca
launch msne 3 4 rev_consensus.py msne
launch msc 3 4 rev_baselines.py multiscale
launch meta 2 6 rev_baselines.py metavis
tmux new-session -d -s tsne_rev_indep "for d in Rabies Genome Dengue ShortRead CoVNCBI Spike7k Host InfluenzaHA BreastCancer LungCancer; do echo \$d; done | xargs -P 5 -I{} $P $C/rev_indep.py {} --threads 4 > logs/rev_indep.log 2>&1; echo ALL_FINISHED >> logs/rev_indep.log"
tmux new-session -d -s tsne_rev_case "$P $C/rev_case.py > logs/rev_case.log 2>&1"
tmux new-session -d -s tsne_rev_scale "bash scripts/fetch_scale.sh data/raw 200000 > logs/rev_fetch.log 2>&1; $P $C/rev_scale.py data/raw/spike_scale.fasta --threads 16 > logs/rev_scale.log 2>&1; echo ALL_FINISHED >> logs/rev_scale.log"
tmux new-session -d -s tsne_rev_valid "for b in CoVNCBI Rabies Spike7k Host; do $P $C/rev_scale.py \$b Spike2Vec --validate --threads 8; done > logs/rev_valid.log 2>&1; echo ALL_FINISHED >> logs/rev_valid.log"
tmux ls | grep tsne_rev
