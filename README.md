### Working Flow
1. Data Preparation.
   * S protein FASTA seq`data/YP_009724390.1.txt`.
   * Variant of concern `data/总结 - 20211201.xlsx`.
2. Find [protein family](http://www.ebi.ac.uk/interpro/result/InterProScan/iprscan5-R20210917-073330-0879-28690888-p2m/).
      * Fasta seqs: `data/protein-matching-IPR042578.fasta`.
      * Detail information `data/protein-matching-IPR042578.json` .
3. Seqs of same protein family are too much, so filter seqs with same name and save`filter_fanmily_protein.py`.
      * Fasta seqs: `data/protein-matching-IPR042578.filter.fasta`.
      * Detail information `data/protein-matching-IPR042578.filter.csv`.
4. `clustalX2` MSA, and export as `data/protein-matching-IPR042578.filter.fasta.aligned`.
5. `cal_procon.py` calculates the conservation score, and result is  `data/procon/type{}.txt`.
   * Due to the default output format is hard to analysis, it will be converted as csv file`data/procon/type{}_parse.csv`.
   
7. `analysis_procon.py` finds the conservation of the variants, and save as `data/procon/analysis.json`.
8. `v2/output_procon_analysis` will construct the network and analysis the network.


### Update log
---
2022.5.16
1. Backup the history result in data/v1/
2. Update the variants `总结 - 20220516`
3. Run `analysis_procon.py` 

---
2022.6.8.

Via reading the result of network, stability and affinity, find the relationship within them.

Network medium files:
* group distribution statistic information.xlsx: variants result
  * 1/4 1/2 3/4 quantile: quantile   
  * mean: average value
  * score: variant score
  * t p: mean value of T-test and p
  * result
  * name: variant name
  * index 
* aas_info.csv: network characteristics of postions


---
2022.7.9
ecdc doesn't provide complete mutations which need to be required
* https://www.ecdc.europa.eu/en/covid-19/variants-concern

Mutation source:
1. https://cov-lineages.org/ can query variant
2. https://outbreak.info/situation-reports?pango=B.1.617.2 can query mutation
   * url to get mutations: https://api.outbreak.info/genomics/lineage-mutations?pangolin_lineage=B.1.617.2&frequency=0.75

Flow
1. Update variant: https://www.ecdc.europa.eu/en/covid-19/variants-concern
2. Crawler mutation `crawler.py`
3. Filter data, remove:
    * duplication
    * without evidence

Network color
* Node
    * mutation #bf643b size 20
    * normal #008ea0  size 10
* Edge
  * neighbour #f66b0e
  * mutation #ffc300
  * normal #b7e5dd

---
2022.8.4. 
1. Gephi 
   * Only 2 color for edges
     * path and not path
     * ♥ mutation and no-mutation
   * position size: conservation score

---
...

---
2022.9.28
* Update network in Gephi
* Calculate DeepDGG for 6VXX and 6VYB
  * Try to submit part of mutation, but failed
  * So, calculate all possible mutations
      * 6VXX: http://protein.org.cn/jobstat.php?type=ddg&id=47029479
      * 6VYB: http://protein.org.cn/jobstat.php?type=ddg&id=87342993