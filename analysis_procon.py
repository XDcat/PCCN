# -*- coding:utf-8 -*-
'''
__author__ = 'XD'
__mtime__ = 2021/9/21
__project__ = Cov2_protein
Fix the Problem, Not the Blame.
'''
import os
import json
import logging
import logconfig
import pandas as pd
from Bio import SeqIO
import itertools

logconfig.setup_logging()
log = logging.getLogger("cov2")


def load_procon_res(res_path):
    d1, d2, d3 = [pd.read_csv(i) for i in res_path]
    # type1
    r1 = {}
    for i, row in d1.iterrows():
        k = int(row.position[:-1])
        r1[k] = row.to_dict()
    yield r1

    # type2
    r2 = {}
    for i, row in d2.iterrows():
        k = tuple(sorted([int(row.site1[:-1]), int(row.site2[:-1])]))
        r2[k] = row.to_dict()
    yield r2

    # type3
    r3 = {}
    for i, row in d3.iterrows():
        k = tuple(sorted([int(row.site1[:-1]), int(row.site2[:-1]), int(row.site3[:-1])]))
        r3[k] = row.to_dict()

    yield r3


def check_aa(seq, aa):
    position = int(aa[1:-1])
    origin = aa[0].upper()
    if seq[position - 1].upper() == origin:
        return True
    else:
        return False


if __name__ == '__main__':
    # 加载变异
    variation_file = "./data/总结 - 2021年9月21日.xlsx"
    log.info("加载变异")
    variation = pd.read_excel(variation_file, sheet_name=1, index_col=0)
    variation["Year and month first detected"] = variation["Year and month first detected"].dt.strftime('%B %d, %Y')
    log.debug("columns: %s", variation.columns)
    log.debug(variation)

    # 检查变异是否合法
    fasta_file = "./data/YP_009724390.1.txt"
    fasta = next(SeqIO.parse(fasta_file, "fasta")).seq
    t_fasta = {}
    for i, row in variation.iterrows():
        aas = row.iloc[0].split(",")
        aas = [i.strip() for i in aas]
        for aa in aas:
            if check_aa(fasta, aa):
                log.debug("%s: %s 合法", i, aa)
            else:
                raise RuntimeError("变异不合法")
        t_fasta[i] = row.to_dict()
        t_fasta[i]["aas"] = aas
    fasta = t_fasta
    log.info("变异均合法")

    # 加载 procon 结果
    result_dir = "./data/procon"
    result_files = [os.path.join(result_dir, "type{}_parse.csv".format(i)) for i in range(1, 4)]
    log.debug("将 procon 结果解析为字典")
    pr = load_procon_res(result_files)

    log.info("解析 type1")
    pr1 = next(pr)
    log.debug("type1: %s", pr1)
    for i, row in fasta.items():
        aas = row["aas"]
        fasta[i]["type1"] = []
        for aa in aas:
            pst = int(aa[1:-1])
            res = pr1.get(pst, None)
            if res is not None:
                fasta[i].get("type1", ).append(res)
    log.debug("type1 result: %s", fasta)

    log.info("解析 type2")
    pr2 = next(pr)
    log.debug("type2: %s", pr2)
    for i, row in fasta.items():
        aas = row["aas"]
        fasta[i]["type2"] = []
        for aa2 in itertools.combinations(aas, 2):
            aa2 = tuple(sorted([int(a[1:-1]) for a in aa2]))
            res = pr2.get(aa2, None)
            if res is not None:
                fasta[i].get("type2").append(res)
    log.debug("type2 result: %s", fasta)

    log.info("解析 type3")
    pr3 = next(pr)
    for i, row in fasta.items():
        aas = row["aas"]
        fasta[i]["type3"] = []
        for aa3 in itertools.combinations(aas, 3):
            aa3 = tuple(sorted([int(a[1:-1]) for a in aa3]))
            res = pr3.get(aa3, None)
            if res is not None:
                fasta[i].get("type3").append(res)
    log.debug("type3 result: %s", fasta)

    log.info("保存解析结果")
    analysis_res_path = "./data/procon/analysis.json"
    with open(analysis_res_path, "w") as f:
        f.write(json.dumps(fasta, indent="\t"))


