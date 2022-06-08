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
   
2. 使用 graphi 进行绘图
   * 使用 `output_gephi_data.py` 导出需要指定格式的文件到`tools`目录中
### 处理流程
1. 准备的数据
   * 找到了 s 蛋白的序列，保存为 `data/YP_009724390.1.txt`
   * 重要的变异 `data/总结 - 20211201.xlsx`
2. 找到 [protein family](http://www.ebi.ac.uk/interpro/result/InterProScan/iprscan5-R20210917-073330-0879-28690888-p2m/)，导出数据
      * fasta 序列：`data/protein-matching-IPR042578.fasta`
      * 名称等详细数据：`data/protein-matching-IPR042578.json` 
3. protein family 蛋白质太多，`filter_fanmily_protein.py`过滤掉同名（物种）的蛋白质，并保存
      * fasta 格式：`data/protein-matching-IPR042578.filter.fasta`
      * 详细数据：`data/protein-matching-IPR042578.filter.csv`
4. 使用`clustalX2`对齐序列，导出为 `data/protein-matching-IPR042578.filter.fasta.aligned`
5. 使用`cal_procon.py`计算保守性，结果为 `data/procon/type{}.txt`。由于输出格式是默认的，不方便后续分析数据，
   将数据解析为csv格式，结果为`data/procon/type{}_parse.csv`
   
6. `analysis_procon.py`找出相关的变异的保守性，并结果为 `data/procon/analysis.json`, 将excel中的数据保存为json文件，之后将可以不用访问json
7. `v2/output_procon_analysis`将上一步得到的结果，解析为excel、图片格式

---
2022年5月16日更新
1. 以前的数据结果都放在了 data/v1/ 中
2. 更新变体文件 `总结 - 20220516`
3. 运行 `analysis_procon.py`

2022年6月8日
通过读取网络的中间文件和stability affinity的结果，寻找其中关系；

中间文件
* group distribution statistic information.xlsx 变体结果
  * 1/4 1/2 3/4 quantile 分位数
  * mean 采样数据平均数
  * score variant 分数
  * t p 计算一组分数的平均值的T-test，以及对应的 p 值
  * result 根据 t p 判断方向
  * name variant 名称
  * index 编号
* aas_info.csv 位点的网络属性
