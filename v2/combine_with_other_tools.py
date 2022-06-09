import json
import math
import re
from functools import reduce
from scipy.stats import pearsonr, spearmanr

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
        def read_avg_shortest_length(file="../data/procon/threshold_100/averrage shortest length.csv"):
            asl = pd.read_csv(file)
            asl["sample avg"] = asl.apply(lambda x: np.mean(json.loads(x.sample_avg_shortest_length)), axis=1)
            asl = asl.loc[:, ["aa", "aas_avg_shortest_length", "sample avg"]]
            asl.columns = ["aa", "average shortest length in variant", "average shortest length in sampled data"]
            return asl

        pcn = self.pcn
        mutation_group = self.mutation_groups
        type1 = pcn.type1
        nd_aas = mutation_group.non_duplicated_aas

        # 默认的数据
        pst_info = pcn._collect_mutation_info(save=False)
        pst_info = pst_info.reset_index(drop=True)
        # average shortest length
        asl = read_avg_shortest_length()
        pst_info = pd.merge(pst_info, asl, left_on="aa", right_on="aa")

        aas_info = pd.DataFrame({"name": nd_aas})
        aas_info["aa"] = aas_info["name"].apply(mutation_group.aa2position)
        aas_info = pd.merge(aas_info, pst_info)
        return aas_info

    def analysis_single_mutation(self):
        bfe = self.read_topnettree()
        ddg = self.read_deep_ddg()
        conservation = self.read_procon_single_mutation()
        conservation_columns = conservation.columns.tolist()
        conservation_data_columns = conservation.columns[2:].tolist()

        # 单 mutation 分析
        info_list = [conservation, bfe, ddg]
        aa_info: pd.DataFrame = pd.concat([i.set_index("name") for i in info_list], axis=1)
        aa_info = aa_info.loc[:, conservation_columns[1:] + ["BFE", "ddG"]]
        aa_info = aa_info.rename(columns={"ddG": "stability", })
        aa_info = aa_info.sort_values("conservation", ascending=True)
        aa_info.to_csv(os.path.join(data_dir, "combine_with_other_tools - single.csv"))
        # 判断数据是否为空值
        # has_conservation = aa_info["conservation"] != 0
        # has_stability = aa_info["stability"].notnull()
        # has_BFE = aa_info["BFE"].notnull()
        # aa_info_no_null = aa_info[pd.DataFrame([has_conservation, has_BFE, has_stability]).all()]
        # aa_info_no_null.to_csv(os.path.join(data_dir, "combine_with_other_tools (no null).csv"))

        # 相关性分析
        c1_columns = ["stability", "BFE"]  # 比较对象 1
        c2_columns = conservation_data_columns  # 比较对象 2
        correlation = self.cal_correlation(aa_info, c1_columns, c2_columns, "single-info")
        correlation.to_csv(os.path.join(data_dir, "combine_with_other_tools - single - correlation.csv"))
        return aa_info, correlation

    def read_procon_co_mutation(self, ):
        pcn = self.pcn
        mutation_group = self.mutation_groups
        G: networkx.Graph = pcn.G
        aa_groups = mutation_group.aa_groups
        aa_groups_info = mutation_group.aa_groups_info

        # edge centrality
        with open(os.path.join("../data/procon/threshold_100", "edge betweenness centrality.json"), ) as f:
            log.info("加载文件")
            edge_betweenness_centrality = json.loads(f.read())

        co_info = []
        for i, grp in enumerate(aa_groups):
            name = aa_groups_info.iloc[i]["name"]
            for a1, a2 in combinations(grp, 2):
                p1 = mutation_group.aa2position(a1)
                p2 = mutation_group.aa2position(a2)
                if G.has_edge(p1, p2):
                    coconservation = G.edges[p1, p2]["weight"]
                    if p1 in edge_betweenness_centrality and p2 in edge_betweenness_centrality[p1]:
                        ebc = edge_betweenness_centrality[p1][p2]
                    elif p2 in edge_betweenness_centrality and p1 in edge_betweenness_centrality[p2]:
                        ebc = edge_betweenness_centrality[p2][p1]
                    else:
                        print("{} {}无 edge centrality".format(a1, a2))
                        ebc = 0

                    co_info.append([name, a1, a2, coconservation, ebc])
        co_info = pd.DataFrame(co_info, columns=["name", "a1", "a2", "co-conservation", "edge centrality"])
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
        data_columns = ["co-conservation", "edge centrality", "BFE", "stability"]
        info = info.drop(["name"], axis=1)  # 不保留毒株名
        info = info.drop_duplicates(["a1", "a2"])  # 删除重复
        info = info.reset_index(drop=True)  # 重设索引

        # info = info[info.loc[:, data_columns].notnull().all(axis=1)]
        # info = info[info.loc[:, data_columns].notnull().all(axis=1)]

        # 相关性分析
        c1_columns = data_columns[2:]
        c2_columns = data_columns[:2]
        correlation = self.cal_correlation(info, c1_columns, c2_columns, "co-info")
        info.to_csv(os.path.join(data_dir, "combine_with_other_tools - co.csv"))
        correlation.to_csv(os.path.join(data_dir, "combine_with_other_tools - co - correlation.csv"))
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

    def analysis_variant(
            self,
            f_single_info=os.path.join(data_dir, "combine_with_other_tools - single.csv"),
            f_co_info=os.path.join(data_dir, "combine_with_other_tools - co.csv")
    ):
        """
        通过读取前两者的输出文件，计算 variant
        :return:
        """
        # middle info
        single_info = pd.read_csv(f_single_info)
        co_info = pd.read_csv(f_co_info, index_col=0)
        # variant info
        mutation_group = self.mutation_groups
        aa_groups = mutation_group.aa_groups
        aa_groups_info = mutation_group.aa_groups_info
        group_info = pd.concat(map(pd.Series, aa_groups), keys=aa_groups_info["name"].to_list()).reset_index()
        group_info.columns = ["variant", "num", "name"]
        # single
        variant_single = pd.merge(group_info, single_info)
        variant_single_group = variant_single.groupby("variant").mean()
        variant_single_group.to_csv(os.path.join(data_dir, "combine_with_other_tools - variant - single.csv"))

        c1_columns = variant_single_group.columns[-2:].to_list()
        c2_columns = variant_single_group.columns[2:-2].to_list()
        single_correlation = self.cal_correlation(variant_single_group, c1_columns, c2_columns, "variant-single")
        single_correlation.to_csv(
            os.path.join(data_dir, "combine_with_other_tools - variant - single - correlation.csv"))

        # co
        co_data_columns = co_info.columns[-4:].tolist()
        variant_co = []
        for i, grp in enumerate(aa_groups):
            name = aa_groups_info.iloc[i]["name"]
            for a1, a2 in combinations(grp, 2):
                idx = (co_info[["a1", "a2"]] == [a1, a2]) | (co_info[["a1", "a2"]] == [a2, a1])
                idx = idx.all(axis=1)
                if sum(idx) > 0:
                    # 存在数据
                    row = co_info[idx].iloc[0]
                    row["name"] = name
                    variant_co.append(row)
        variant_co = pd.DataFrame(variant_co).reset_index(drop=True)
        variant_co = variant_co.groupby("name").mean()
        single_correlation.to_csv(os.path.join(data_dir, "combine_with_other_tools - variant - co.csv"))

        c1_columns = co_data_columns[2:]
        c2_columns = co_data_columns[:2]
        single_correlation = self.cal_correlation(variant_co, c1_columns, c2_columns, "variant-c0")
        single_correlation.to_csv(os.path.join(data_dir, "combine_with_other_tools - variant - co - correlation.csv"))

    @staticmethod
    def cal_correlation(info: pd.DataFrame, c1_columns: List, c2_columns: List, name: str):
        """
        # 相关性分析
        :param info:  数据源
        :param c1_columns: 对比对象1
        :param c2_columns: 对比对象2
        :param name: 结果名称
        :return: pd.DataFrame
        """
        correlation = []
        for c1 in c1_columns:
            for c2 in c2_columns:
                compare_data = info.loc[:, [c1, c2]]
                # 去除空数据（异常值）的影响
                compare_data = compare_data[compare_data.notna().all(axis=1)]
                c1_data = compare_data.loc[:, c1].to_list()
                c2_data = compare_data.loc[:, c2].to_list()
                print("总共数据量{}, 非空数据量{}".format(info.shape[0], compare_data.shape[0]))

                # 皮尔逊相关系数
                p = pearsonr(c1_data, c2_data)
                print(f"{c1} / {c2} : pearsonr={p[0]}, p={p[1]}")
                # spear
                sp = spearmanr(c1_data, c2_data)
                print(f"{c1} / {c2} : spearmanr={sp[0]}, p={sp[1]}")
                # 结果
                correlation.append([c1, c2, p[0], p[1], sp[0], sp[1], info.shape[0], compare_data.shape[0]])

        correlation = pd.DataFrame(
            correlation,
            columns=["name1", "name2", "pearsonr correlation", "pearsonr p", "spearmanr correlation", "spearmanr p",
                     "total data", "not null data"]
        )
        correlation.name = name
        return correlation

    @staticmethod
    def parse_result(
            f_single_correlation=os.path.join(data_dir, "combine_with_other_tools - single - correlation.csv"),
            f_co_correlation=os.path.join(data_dir, "combine_with_other_tools - co - correlation.csv"),
            f_variant_s_correlation=os.path.join(data_dir,
                                                 "combine_with_other_tools - variant - single - correlation.csv"),
            f_variant_co_correlation=os.path.join(data_dir,
                                                  "combine_with_other_tools - variant - co - correlation.csv")

    ):
        """分析三种类别的结果文件，并绘制成表格和热力图"""
        names = ["substitution", "co-substitution", "variant (single)", "variant (co)"]
        correlations = [pd.read_csv(i, index_col=[1, 2]) for i in
                        [f_single_correlation, f_co_correlation, f_variant_s_correlation, f_variant_co_correlation]]
        all_cor = pd.concat(correlations, keys=names)

        # 统计两种相关性的个数
        # spearmanr 比较多
        count_p = sum(all_cor["pearsonr p"] < 0.05)
        count_sp = sum(all_cor["spearmanr p"] < 0.05)
        print(f"总个数{all_cor.shape[0]}, pearson 有效个数{count_p}, spearsanr 有效个数{count_sp}")



        def draw_heatmap(plot_cor):
            # 热力图
            fig: plt.Figure
            axes: List[plt.Axes]
            fig, axes = plt.subplots(1, 2, sharey="row", figsize=(7, 7))
            for i, ax in zip(target_columns, axes[:2]):
                target_data = plot_cor[i].unstack(level=1)
                sns.heatmap(target_data, ax=ax)
                ax.set_xlabel(i)
                ax.set_ylabel("")
            fig.tight_layout()
            # fig.show()
            return fig

        # 找到对应数据
        target = "spearmanr"
        target_columns = [target + " correlation", target + " p"]
        # 整体图
        plot_cor = all_cor
        fig = draw_heatmap(plot_cor)
        fig.show()
        fig.savefig(os.path.join(data_dir, "combine_with_other_tools - parse - heatmap1.png"))
        # 有显著性数据的图
        plot_cor = all_cor
        idx_invalid = plot_cor[target_columns[-1]] >= 0.05
        plot_cor[target_columns[0]][idx_invalid] = pd.NA
        plot_cor[target_columns[1]][idx_invalid] = pd.NA
        fig = draw_heatmap(plot_cor)
        fig.show()
        fig.savefig(os.path.join(data_dir, "combine_with_other_tools - parse - heatmap2.png"))

        # 找出有显著性的数据表格
        # 单独为 BFE 和 stability 添加一个表格
        valid_cor = all_cor[all_cor[target_columns[-1]] < 0.05]
        valid_cor = valid_cor.swaplevel(0, 1).sort_index(level=0)
        valid_cor = valid_cor[target_columns]
        valid_cor.index.names = ["#", "kind", "network characteristic"]
        valid_cor.columns = ["spearmanr correlation", "p value"]
        valid_cor.to_csv(os.path.join(data_dir, "combine_with_other_tools - parse.csv"))

if __name__ == '__main__':
    # mutation_groups = AnalysisMutationGroup()
    # pcn = ProConNetwork(mutation_groups, threshold=100)
    # cr = CombineResult()

    # 分析单个位点的数据
    # singe_info = cr.analysis_single_mutation()

    # 分析两个位点
    # co_info = cr.analysis_co_mutations()

    # 分析毒株
    # cr.analysis_variant()
    CombineResult.parse_result()
