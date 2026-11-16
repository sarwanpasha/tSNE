#!/bin/bash
# Download up to N SARS-CoV-2 spike (surface glycoprotein) protein sequences of full length from NCBI Protein
# for the scalability experiment. usage: fetch_scale.sh OUTDIR [N]
OUT=${1:-workspace/data/raw}; N=${2:-200000}; mkdir -p $OUT; E=https://eutils.ncbi.nlm.nih.gov/entrez/eutils
Q='SARS-CoV-2[Organism]+AND+surface+glycoprotein[Protein+Name]+AND+1260:1290[SLEN]'
R=$(curl -gs "$E/esearch.fcgi?db=protein&term=$Q&usehistory=y&retmax=0")
WE=$(echo "$R" | grep -o '<WebEnv>[^<]*' | cut -c9-); QK=$(echo "$R" | grep -o '<QueryKey>[^<]*' | cut -c11-)
echo "count: $(echo "$R" | grep -o '<Count>[^<]*' | head -1 | cut -c8-)"
: > $OUT/spike_scale.fasta
for ((s=0; s<N; s+=10000)); do
  curl -gs "$E/efetch.fcgi?db=protein&query_key=$QK&WebEnv=$WE&rettype=fasta&retmode=text&retstart=$s&retmax=10000" >> $OUT/spike_scale.fasta
  echo "fetched $s"; sleep 1
done
grep -c '>' $OUT/spike_scale.fasta
