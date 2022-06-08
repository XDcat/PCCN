import json
import math
import re
from functools import reduce
from scipy.stats import pearsonr

from pyecharts import options as opts
from pyecharts.charts import Graph
import networkx
import seaborn as sns
import time
from Bio import SeqIO
import networkx as nx
from networkx.algorithms.centrality import degree_centrality, betweenness_centrality, closeness_centrality, \
    edge_betweenness_centrality
from networkx.algorithms.shortest_paths import shortest_path_length
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
from scipy.special import comb
from scipy.stats import mannwhitneyu, ttest_1samp
from itertools import combinations, permutations
from collections import defaultdict
from brokenaxes import brokenaxes
from matplotlib.gridspec import GridSpec
from sklearn import preprocessing
# 组合数
from scipy.special import comb
# 日志
import logging
import logconfig
import pickle
from typing import List

logconfig.setup_logging()
log = logging.getLogger("cov2")

AA = ['-', 'C', 'F', 'I', 'L', 'M', 'V', 'W', 'Y', 'A', 'T', 'D', 'E', 'G', 'P', 'N', 'Q', 'S', 'H', 'R', 'K']
# origin_aas = ['P681R', 'Q954H', 'Q613H', 'Y449H', 'Y505H', 'G446S', 'N764K', 'R408S', 'A67V', 'S494P', 'G142D', 'N969K',
#               'T547K', 'E516Q', 'E484A', 'N440K', 'K417T', 'V213G', 'Y145H', 'N439K', 'P384L', 'N211I', 'N679K',
#               'D405N', 'L452Q', 'Q677H', 'F490S', 'D796Y', 'A222V', 'L981F', 'T478K', 'T95I', 'V367F', 'N501Y', 'S477N',
#               'N501T', 'T376A', 'H655Y', 'G339D', 'A653V', 'F486V', 'D614G', 'P681H', 'L452X', 'N856K', 'S373P',
#               'K417N', 'S371F', 'S375F', 'S371L', 'Q493R', 'E484K', 'Q498R', 'E484Q', 'R346K', 'A701V', 'L452R',
#               'G496S']
from output_procon_analysis import AnalysisMutationGroup, ProConNetwork

analysisMutationGroup = AnalysisMutationGroup()
origin_aas = analysisMutationGroup.non_duplicated_aas

# sns.set()
sns.set_style("ticks")  # 主题: 白色背景且有边框

data_dir = "../data/procon"


class CombineResult:
    def __init__(self, ):
        self.mutation_groups = AnalysisMutationGroup()
        self.pcn = ProConNetwork(self.mutation_groups, threshold=100)

    def read_procon_single_mutation(self, ):
        pcn = self.pcn
        mutation_group = self.mutation_groups
        type1 = pcn.type1
        nd_aas = mutation_group.non_duplicated_aas

        pst_info = pcn._collect_mutation_info(save=False)
        pst_info = pst_info.reset_index(drop=True)
        aas_info = pd.DataFrame({"name": nd_aas})
        aas_info["aa"] = aas_info["name"].apply(mutation_group.aa2position)
        aas_info = pd.merge(aas_info, pst_info)
        return aas_info

    def analysis_single_mutation(self):
        bfe = self.read_topnettree()
        ddg = self.read_deep_ddg()
        conservation = self.read_procon_single_mutation()

        # 单 mutation 分析
        info_list = [conservation, bfe, ddg]
        aa_info: pd.DataFrame = pd.concat([i.set_index("name") for i in info_list], axis=1)
        aa_info = aa_info.loc[:, ["aa", "normalized conservation", "BFE", "ddG"]]
        aa_info = aa_info.rename(columns={"ddG": "stability", "normalized conservation": "conservation"})
        aa_info = aa_info.sort_values("conservation", ascending=True)
        # aa_info.to_csv(os.path.join(data_dir, "combine_with_other_tools.csv"))
        # 判断数据是否为空值
        has_conservation = aa_info["conservation"] != 0
        has_stability = aa_info["stability"].notnull()
        has_BFE = aa_info["BFE"].notnull()
        aa_info_no_null = aa_info[pd.DataFrame([has_conservation, has_BFE, has_stability]).all()]
        aa_info_no_null.to_csv(os.path.join(data_dir, "combine_with_other_tools (no null).csv"))

        # 相关性分析
        data_columns = ["conservation", "stability", "BFE"]
        correlation = []
        for (name1, column1), (name2, column2) in combinations(list(aa_info_no_null.loc[:, data_columns].iteritems()),
                                                               2):
            p = pearsonr(column1, column2)
            print(f"{name1} / {name2} : pearsonr={p[0]}, p={p[1]}")
            correlation.append([name1, name2, p[0], p[1]])
        correlation = pd.DataFrame(correlation, columns=["name1", "name2", "correlation", "p"])
        correlation.name = "single-info"
        return aa_info_no_null, correlation

    def read_procon_co_mutation(self, analysis_file="../data/procon/analysis.json"):
        pcn = self.pcn
        mutation_group = self.mutation_groups
        G: networkx.Graph = pcn.G
        aa_groups = mutation_group.aa_groups
        aa_groups_info = mutation_group.aa_groups_info

        co_info = []
        for i, grp in enumerate(aa_groups):
            name = aa_groups_info.iloc[i]["name"]
            for a1, a2 in combinations(grp, 2):
                p1 = mutation_group.aa2position(a1)
                p2 = mutation_group.aa2position(a2)
                if G.has_edge(p1, p2):
                    co_info.append([name, a1, a2, G.edges[p1, p2]["weight"],])
        co_info = pd.DataFrame(co_info, columns=["name", "a1", "a2", "co-conservation", ])
        # print(co_info)
        return co_info

    def analysis_co_mutations(self):
        bfe = self.read_topnettree()
        bfe = bfe.set_index("name")
        ddg = self.read_deep_ddg()
        ddg = ddg.set_index("name")
        conservation = self.read_procon_co_mutation()

        def cal_bfe(row):
            a1 = row["a1"]
            a2 = row["a2"]
            if a1 in bfe.index and a2 in bfe.index:
                res = (bfe.loc[a1]["BFE"] + bfe.loc[a2]["BFE"]) / 2
                return res
            return None

        def cal_ddg(row):
            a1 = row["a1"]
            a2 = row["a2"]
            if a1 in ddg.index and a2 in ddg.index:
                res = (ddg.loc[a1]["ddG"] + ddg.loc[a2]["ddG"]) / 2
                return res
            return None

        info = conservation
        info = info.sort_values("co-conservation", ascending=True)
        info["BFE"] = info.apply(cal_bfe, axis=1)
        info["stability"] = info.apply(cal_ddg, axis=1)

        # 检查是否为空
        data_columns = ["co-conservation", "BFE", "stability"]
        info = info[info.loc[:, data_columns].notnull().all(axis=1)]

        # 相关性分析
        correlation = []
        for (name1, column1), (name2, column2) in combinations(list(info.loc[:, data_columns].iteritems()),
                                                               2):
            p = pearsonr(column1, column2)
            print(f"{name1} / {name2} : pearsonr={p[0]}, p={p[1]}")
            correlation.append([name1, name2, p[0], p[1]])
        correlation = pd.DataFrame(correlation, columns=["name1", "name2", "correlation", "p"])
        correlation.name = "co-info"
        info.to_csv(os.path.join(data_dir, "combine_with_other_tools co-info.csv"))
        return info, correlation

    def read_topnettree(self, result_file="../tools/try topnettree/data/result.txt"):
        with open(result_file) as f:
            lines = f.readlines()
        lines = lines[1:]
        lines = list(map(lambda x: x.strip(), lines))
        lines = np.reshape(lines, (-1, 2)).tolist()
        bfe = []
        for line in lines:
            name = line[0].split()[0]
            score = re.search("\d+\.\d+", line[1])
            score = score.group() if score else None
            bfe.append({"name": name, "score": score})
            # print(name, score)

        bfe = pd.DataFrame(bfe)
        bfe["score"] = bfe["score"].astype(float)
        bfe["idx"] = bfe["name"].str[1:-1].astype(int)
        bfe = bfe[bfe["score"].notna()]

        bfe = bfe.sort_values("idx")
        bfe = bfe.reset_index(drop=True)
        bfe = bfe.rename(columns={"score": "BFE"})
        return bfe

    def read_deep_ddg(self, result_file="../tools/try DeepDDG/result-QHD43416.ddg"):
        data = pd.read_csv(result_file, delimiter="\s+", skiprows=[0, ], header=None)
        data.columns = "#chain WT ResID Mut ddG".split()
        data["name"] = data.apply(lambda x: "{}{}{}".format(x["WT"], x["ResID"], x["Mut"]), axis=1)
        data.index = data["name"]
        # for i in origin_aas:
        #     if i not in data["name"]:
        #         print(i)
        # print(data)
        # print(data[data["ResID"] == 452])
        # print(data.columns)
        # print(data.dtypes)
        fdata = data.loc[list(filter(lambda x: x[-1] != "X", origin_aas)), :]
        fdata = fdata.sort_values("ResID")
        fdata = fdata.reset_index(drop=True)

        return fdata


if __name__ == '__main__':
    # mutation_groups = AnalysisMutationGroup()
    # pcn = ProConNetwork(mutation_groups, threshold=100)
    cr = CombineResult()

    # 分析单个位点的数据
    singe_info = cr.analysis_single_mutation()

    # 分析两个位点
    co_info = cr.analysis_co_mutations()

    correlation = pd.concat([singe_info[1], co_info[1]], keys=["single", "co"])
    print(correlation)
    correlation.to_csv(os.path.join(data_dir, "combine_with_other_tools correlation.csv"))
