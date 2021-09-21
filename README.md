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

### 错误的尝试
1. 使用 diamond 计算同源序列，但是相似度太高，无法计算出保守型，相关文件如下：
   * `data/diamond_result` diamond 计算结果
   * `get_fasta_seq.py` 下载序列
   * `data/genes`  下载序列的文件夹
### 处理流程
1. 准备的数据
   * 找到了 s 蛋白的序列，保存为 `data/YP_009724390.1.txt`
   * 重要的变异 `data/总结 - 2021年9月21日.xlsx`
2. 找到 [protein family](http://www.ebi.ac.uk/interpro/result/InterProScan/iprscan5-R20210917-073330-0879-28690888-p2m/)，导出数据
      * fasta 序列：`data/protein-matching-IPR042578.fasta`
      * 名称等详细数据：`data/protein-matching-IPR042578.json` 
3. protein family 蛋白质太多，`filter_fanmily_protein.py`过滤掉同名（物种）的蛋白质，并保存
      * fasta 格式：`data/protein-matching-IPR042578.filter.fasta`
      * 详细数据：`data/protein-matching-IPR042578.filter.csv`
4. 使用`clustalX2`对齐序列，导出为 `data/protein-matching-IPR042578.filter.fasta.aligned`
5. 使用`cal_procon.py`计算保守性，结果为 `data/procon/type{}.txt`。由于输出格式是默认的，不方便后续分析数据，
   将数据解析为csv格式，结果为`data/procon/type{}_parse.csv`
   
6. `analysis_procon.py`找出重要的变异的保守性，并保存结果为 csv 文件，方便后续分析,结果为 `data/procon/analysis.json`
