# -*- coding:utf-8 -*-
'''
__author__ = 'XD'
__mtime__ = 2021/9/20
__project__ = Cov2_protein
Fix the Problem, Not the Blame.
'''
from ProCon.myProCon import ProbabilityCalculator, MutualInformation, TripletFinder
from Bio import AlignIO
import os
import pandas as pd

import logging
import logconfig
logconfig.setup_logging()
log = logging.getLogger("cov2")

if __name__ == '__main__':
    align_fasta = r"D:\Box\python\Cov2_protein\data\protein-matching-IPR042578.filter.fasta.aligned"
    result_dir = "./data/procon"
    result_files = [os.path.join(result_dir, "type{}.txt".format(i)) for i in range(1, 4)]

    # 读取原始序列对齐的序列
    align_seqs = AlignIO.parse(align_fasta, "fasta")
    origin_seq = next(align_seqs)[0]
    log.debug(len(origin_seq.seq))
    assert origin_seq.id == "YP_009724390.1"
    # 建立 align 前后索引映射
    # align_map = []
    align_map = {}
    count_na = 0
    for i, j in enumerate(origin_seq.seq):
        if j == "-":
            count_na += 1
        align_map[i + 1] = {"origin": i + 1 - count_na, "aa": j}
    # log.debug("align map:\n%s", align_map)

    # 计算保守性
    log.info("计算保守性")
    if not all([os.path.exists(i) for i in result_files]):
        log.info("开始计算 type1")
        pc = ProbabilityCalculator.ProbabilityCalculator(align_fasta)
        log.info("pc result")
        log.info(pc.get_entropy_0())
        log.info(pc.get_gap_20())
        pc.print_sorted_inf_20()
        pc.inf_to_file_20(result_files[0])

        log.info("开始计算 type2")
        mi = MutualInformation.MutualInformation(pc)
        log.info(mi.get_mut_inf())
        mi.mut_inf_to_file(result_files[1])

        log.info("开始计算 type3")
        tf = TripletFinder.TripletFinder(mi)
        log.info(tf.get_triplets())
        tf.display_triplets()
        tf.tps_to_file(result_files[2])
    else:
        log.info("保守性已经计算完毕，直接复用历史文件")


    # 解析结果数据
    def restore_aa(s, mp):
        """
        将 site 复原，e.g. 1030T -> 900T
        :param s: series 序列
        :param mp: 映射字典
        :return: 复原后的 series
        """

        def aux(site):
            i = int(site[:-1])
            a = site[-1]
            if mp[i]["aa"] == a:
                return f"{mp[i]['origin']}{a}"
            else:
                raise RuntimeError("映射位点时出错 {}".format(site))

        res = s.apply(aux)
        return res


    log.info("解析 type1")
    t1 = pd.read_csv(result_files[0], sep="\s+", skipfooter=1)
    # t1["restore"] = restore_aa(t1["position"], align_map)
    t1["position"] = restore_aa(t1["position"], align_map)
    t1.to_csv(result_files[0][:-4] + "_parse.csv", index=False)

    log.info("解析 type2")
    t2 = pd.read_csv(result_files[1], sep="\s+", ).iloc[:, :-1]
    t2.columns = "rank site1 site2 info".split()
    t2["site1"] = restore_aa(t2["site1"], align_map)
    t2["site2"] = restore_aa(t2["site2"], align_map)
    t2.to_csv(result_files[1][:-4] + "_parse.csv", index=False)

    log.info("解析 type3")
    t3 = pd.read_csv(result_files[2], sep="\s+", )
    t3["site1"] = restore_aa(t3["site1"], align_map)
    t3["site2"] = restore_aa(t3["site2"], align_map)
    t3["site3"] = restore_aa(t3["site3"], align_map)
    t3.to_csv(result_files[2][:-4] + "_parse.csv", index=False)
