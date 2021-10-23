# -*- coding:utf-8 -*-
'''
__author__ = 'XD'
__mtime__ = 2021/9/17
__project__ = Cov2_protein
Fix the Problem, Not the Blame.

http://www.ebi.ac.uk/interpro/result/InterProScan/iprscan5-R20210917-073330-0879-28690888-p2m/
从上述网站，找到了 s 蛋白的 protein family，但是蛋白质数目太多，除去同名的蛋白质，还剩下 261 条，并导出
'''
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import pandas as pd
import json
import logging
import logconfig

logconfig.setup_logging()
log = logging.getLogger("cov2")


def load_fasta(fasta_file):
    seqs = SeqIO.parse(fasta_file, "fasta")
    seqs = [[i.id.split("|")[0], i.id, i.seq] for i in seqs]
    columns = ["id", "desc", "seq", ]
    seqs = pd.DataFrame(seqs, columns=columns)
    return seqs


def load_json(json_file):
    with open(json_file) as f:
        info = json.load(f)
    tinfo = []
    for i in info:
        metadata = i["metadata"]
        metadata.update(metadata.pop("source_organism"))
        tinfo.append(metadata)
    info = pd.DataFrame(tinfo)
    # info = info.rename({"accession": "id"})
    return info


if __name__ == '__main__':
    fasta_file = "./data/protein-matching-IPR042578.fasta"
    json_file = "./data/protein-matching-IPR042578.json"
    query_fasta_file = "./data/YP_009724390.1.txt"
    fasta = load_fasta(fasta_file)
    info = load_json(json_file)
    log.debug("fasta columns: %s", fasta.columns)
    log.debug("fasta shape %s ", fasta.shape)
    log.debug("info columns: %s", info.columns)
    log.debug("info shape %s ", info.shape)
    # 过滤重名
    log.debug("info 中不重复的物种: %s", info.shape[0] - info.fullName.duplicated().sum())
    data = pd.merge(fasta, info, left_on="id", right_on="accession")
    log.debug("合并结果:\n%s", data)
    # data = data[data.fullName.duplicated()]
    data = data.drop_duplicates("fullName")
    log.debug("去重后结果:\n%s", data)

    # 导出 csv 和 fasta 格式
    data.to_csv("./data/protein-matching-IPR042578.filter.csv")
    res_fasta = list(SeqIO.parse(query_fasta_file, "fasta"))
    for i, row in data.iterrows():
        rec = SeqRecord(
            Seq(row.seq),
            id=row.id,
            # description="|".join([row.fullName, row.source_database])
            description="| " + " | ".join([row.source_database, row.fullName, ])
        )
        res_fasta.append(rec)
    SeqIO.write(res_fasta, "./data/protein-matching-IPR042578.filter.fasta", "fasta")
