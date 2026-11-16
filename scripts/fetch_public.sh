#!/bin/bash
# Downloads public datasets used as additional benchmarks. Idempotent.
set -u
# Requires the NCBI 'datasets' CLI on PATH (https://www.ncbi.nlm.nih.gov/datasets/docs/v2/command-line-tools/download-and-install/)
R=${TSNE_ROOT:-workspace}; B=datasets; mkdir -p $R/data/raw && cd $R/data/raw
PY=${PYTHON:-python}
acc() { $PY -c "import sys,json;[print(json.loads(l)['accession']) for l in sys.stdin if l.strip()]"; }

# 1) Anticancer peptides (UCI id 589): breast + lung
[ -f ACPs_Lung_cancer.csv ] || { curl -sL -o acp.zip https://archive.ics.uci.edu/static/public/589/anticancer+peptides.zip && unzip -o -q acp.zip; }
echo "ACP: $(ls ACPs_*csv)"

# 2) Dengue serotypes 1-4, complete genomes (NCBI Virus)
for t in 11053:DENV1 11060:DENV2 11069:DENV3 11070:DENV4; do
  id=${t%:*}; nm=${t#*:}
  [ -s dengue_$nm/ncbi_dataset/data/genomic.fna ] && continue
  $B summary virus genome taxon $id --complete-only --limit 1500 --as-json-lines | acc > dengue_$nm.acc
  $B download virus genome accession --inputfile dengue_$nm.acc --filename dengue_$nm.zip --include genome && unzip -o -q dengue_$nm.zip -d dengue_$nm
  echo "$nm $(grep -c '>' dengue_$nm/ncbi_dataset/data/genomic.fna)"
done

# 3) Influenza A hemagglutinin proteins by subtype (NCBI E-utilities, taxonomy-based query)
E=https://eutils.ncbi.nlm.nih.gov/entrez/eutils
for t in 114727:H1N1 119210:H3N2 102793:H5N1 333278:H7N9 102796:H9N2; do id=${t%:*}; st=${t#*:}
  [ -s flu_$st.faa ] && continue
  q="txid$id[Organism:exp] AND hemagglutinin[Title] AND 540:590[SLEN]"
  curl -s -G "$E/esearch.fcgi" --data-urlencode "db=protein" --data-urlencode "term=$q" --data-urlencode "retmax=1500" | grep -o '<Id>[0-9]*</Id>' | sed 's/<[^>]*>//g' | split -l 300 - fid_$st.
  for f in fid_$st.*; do curl -s "$E/efetch.fcgi?db=protein&rettype=fasta&retmode=text&id=$(paste -sd, $f)" >> flu_$st.faa; sleep 0.5; done; rm -f fid_$st.*
  echo "$st $(grep -c '>' flu_$st.faa)"
done

# 4) SARS-CoV-2 spike proteins with Pango lineage (NCBI Virus) - large-scale / scalability set
for L in B.1.1.7 B.1.617.2 AY.4 BA.1 BA.2 BA.5 XBB.1.5 P.1 B.1.351 BQ.1.1; do
  [ -s cov_$L/ncbi_dataset/data/protein.faa ] && continue
  $B summary virus genome taxon SARS-CoV-2 --lineage $L --complete-only --limit 5000 --as-json-lines | acc > cov_$L.acc
  $B download virus genome accession --inputfile cov_$L.acc --filename cov_$L.zip --include protein && unzip -o -q cov_$L.zip -d cov_$L
  echo "$L $(grep -c 'surface glycoprotein' cov_$L/ncbi_dataset/data/protein.faa)"
done
echo FETCH_DONE
