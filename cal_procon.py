# -*- coding:utf-8 -*-
from ProCon.myProCon import ProbabilityCalculator, MutualInformation, TripletFinder
from Bio import AlignIO, SeqIO
import os
import pandas as pd
import numpy as np

import logging
import logconfig

logconfig.setup_logging()
log = logging.getLogger("cov2")

if __name__ == '__main__':
    align_fasta = r".\data\protein-matching-IPR042578.filter.fasta.aligned"
    origin_fasta_file = r"./data/YP_009724390.1.txt"
    result_dir = "./data/procon"
    result_files = [os.path.join(result_dir, "type{}.txt".format(i)) for i in range(1, 4)]

    # get fasta
    origin_fasta = SeqIO.parse(origin_fasta_file, "fasta")
    origin_fasta = next(origin_fasta)
    origin_fasta_len = len(origin_fasta.seq)
    log.debug("origin fasta len: %s", origin_fasta_len)

    # read msa
    align_seqs = AlignIO.parse(align_fasta, "fasta")
    origin_seq = next(align_seqs)[0]
    log.debug(len(origin_seq.seq))
    assert origin_seq.id == "YP_009724390.1"
    # map
    # align_map = []
    align_map = {}
    count_na = 0
    for i, j in enumerate(origin_seq.seq):
        if j == "-":
            count_na += 1
        align_map[i + 1] = {"origin": i + 1 - count_na, "aa": j}
    # log.debug("align map:\n%s", align_map)

    # cal conservation
    log.info("cal conservation")
    if not all([os.path.exists(i) for i in result_files]):
        log.info("start to cal type1")
        pc = ProbabilityCalculator.ProbabilityCalculator(align_fasta)
        log.info("pc result")
        log.info(pc.get_entropy_0())
        log.info(pc.get_gap_20())
        pc.print_sorted_inf_20()
        pc.inf_to_file_20(result_files[0])

        log.info("start to cal type2")
        mi = MutualInformation.MutualInformation(pc)
        log.info(mi.get_mut_inf())
        mi.mut_inf_to_file(result_files[1])

        log.info("start to cal type3")
        tf = TripletFinder.TripletFinder(mi)
        log.info(tf.get_triplets())
        tf.display_triplets()
        tf.tps_to_file(result_files[2])
    else:
        log.info("use history filse directly")


    # parse
    def restore_aa(s, mp):
        def aux(site):
            i = int(site[:-1])
            a = site[-1]
            if mp[i]["aa"] == a:
                return f"{mp[i]['origin']}{a}"
            else:
                raise RuntimeError("error {}".format(site))

        res = s.apply(aux)
        return res


    log.info("parse type1")
    t1 = pd.read_csv(result_files[0], sep="\s+", skipfooter=1)
    # t1["restore"] = restore_aa(t1["position"], align_map)
    t1 = t1[t1["position"].apply(lambda x: x[-1] != "-")]  # filter -
    t1["position"] = restore_aa(t1["position"], align_map)
    t1["rank"] = np.arange(t1.shape[0]) + 1  # rank
    t1["rate"] = t1["rank"] / origin_fasta_len * 100
    # t1.to_csv(result_files[0][:-4] + "_parse.csv", index=False)

    log.info("parse type2")
    t2 = pd.read_csv(result_files[1], sep="\s+", ).iloc[:, :-1]
    t2.columns = "rank site1 site2 info".split()
    t2 = t2[t2["site1"].apply(lambda x: x[-1] != "-") & t2["site2"].apply(lambda x: x[-1] != "-")]
    t2["site1"] = restore_aa(t2["site1"], align_map)
    t2["site2"] = restore_aa(t2["site2"], align_map)
    n = origin_fasta_len ** 2
    t2["rate"] = t2["rank"] / n * 100
    # t2.to_csv(result_files[1][:-4] + "_parse.csv", index=False)

    log.info("parse type3")
    t3 = pd.read_csv(result_files[2], sep="\s+", )
    t3["site1"] = restore_aa(t3["site1"], align_map)
    t3["site2"] = restore_aa(t3["site2"], align_map)
    t3["site3"] = restore_aa(t3["site3"], align_map)
    # t3.to_csv(result_files[2][:-4] + "_parse.csv", index=False)
