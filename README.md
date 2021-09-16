### diamond
wiki https://github.com/bbuchfink/diamond/wiki
How to use?
1. download nr db from blast https://ftp.ncbi.nlm.nih.gov/blast/db/v5/FASTA/nr.gz
2. creating a diamond-formatted database file 
    ```shell
    ./diamond makedb --in nr -d nr.dmnd
    ```
3. running a search in blastp mode
   ```shell
   ./diamond blastp -d  nr.dmnd -q queries.fasta -o matches.tsv
   ```