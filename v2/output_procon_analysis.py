import json
# 日志
import logging
import math
import os
import pickle
import re
import time
from collections import defaultdict
from itertools import combinations, permutations
from typing import List

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from Bio import SeqIO
from networkx.algorithms.centrality import degree_centrality, betweenness_centrality, closeness_centrality, \
    edge_betweenness_centrality
from networkx.algorithms.shortest_paths import shortest_path_length
from pyecharts import options as opts
from pyecharts.charts import Graph
# 组合数
from scipy.special import comb
from scipy.stats import mannwhitneyu, ttest_1samp
from sklearn import preprocessing

import logconfig

logconfig.setup_logging()
log = logging.getLogger("cov2")

AA = ['-', 'C', 'F', 'I', 'L', 'M', 'V', 'W', 'Y', 'A', 'T', 'D', 'E', 'G', 'P', 'N', 'Q', 'S', 'H', 'R', 'K']

# sns.set()
sns.set_style("ticks")  # 主题: 白色背景且有边框
# 更新字体大小
plt.rcParams.update({'font.size': 16})
plt.rcParams["axes.titlesize"] = "medium"


# 归一化
def scaler(d: dict):
    min_max_scaler = preprocessing.MinMaxScaler()
    keys = d.keys()
    # values = np.array(d.values(), dtype=float)
    values = np.array(list(d.values())).reshape(-1, 1)  # 变为 array
    values = min_max_scaler.fit_transform(values)  # 归一化
    values = values.reshape(-1).tolist()  # 恢复为列表
    res = dict(zip(keys, values))  # 恢复为字典
    return res


class AnalysisMutationGroup:
    def __init__(self, analysis="../data/procon/analysis.json", seed=0, fasta_file="../data/YP_009724390.1.txt", ):
        # 关注的变异组的相关数据
        with open(analysis) as f:
            self.analysis: dict = json.load(f)

        self.seed = seed
        # 氨基酸序列以及所有位点
        self.fasta = next(SeqIO.parse(fasta_file, "fasta")).seq
        self.positions = [f"{i + 1}{aa.upper()}" for i, aa in enumerate(self.fasta)]  # 所有可能的位点
        # 变异组以及对应位点
        self.aa_groups, self.aa_groups_info = self.get_aa_groups()
        self.aa_groups_position = self.get_aa_groups_position()
        # 根据变异的数目采样: 直接使用 group sample 中的数据
        self.group_count_sample = self.resample_groups()

        # 不重复变异以及对应位点
        self.non_duplicated_aas = self.get_non_duplicated_aas()
        self.non_duplicated_aas_positions = self.get_non_duplicated_aas_position()
        self.non_duplicated_aas_sample = self.resample_aas(self.non_duplicated_aas_positions)

        self.display_seq_and_aa()

    @staticmethod
    def aa2position(aa: str):
        if aa[0] in AA and aa[-1] in aa and aa[1:-1].isdigit():
            return aa[1:-1] + aa[0]
        elif aa[:-1].isdigit() and aa[-1] in aa:
            return aa
        else:
            raise RuntimeError(f"无法转化aa={aa}")

    def get_aa_groups(self):
        aas = []  # 变异
        names = []  # 名称
        categroies = []  # 类别
        for i, row in self.analysis.items():
            aas.append(row["aas"])
            categroies.append(row["WHO label"])
            # categroies.append(row["Category"])
            if type(row["WHO label"]) == str:
                # name = "{} ({})".format(row["Lineage + additional mutations"], row["WHO label"])
                name = "{}({})".format(row["Lineage + additional mutations"], row["WHO label"])
            else:
                name = "{}".format(row["Lineage + additional mutations"])
            names.append(name)

        info = pd.DataFrame({"name": names, "category": categroies})
        # 为类别设置颜色
        info["color"] = sns.hls_palette(n_colors=len(info), )
        return aas, info

    def get_aa_groups_position(self):
        groups, _ = self.get_aa_groups()
        groups = [[self.aa2position(aa) for aa in group] for group in groups]
        return groups

    def get_non_duplicated_aas(self):
        aas = []
        for i, row in self.analysis.items():
            aas += row["aas"]
        aas = list(set(aas))
        return aas

    def get_non_duplicated_aas_position(self):
        aas = self.get_non_duplicated_aas()
        positions = [self.aa2position(aa) for aa in aas]
        positions = list(set(positions))
        return positions

    def resample_aas(self, aas, N=1000):
        # 重采样
        positions = pd.Series(self.positions)
        log.debug("开始采样，采样数目为%s", N)
        sample_aas = [positions.sample(n=len(aas), random_state=self.seed + i).tolist() for i in range(N)]
        return sample_aas

    def resample_groups(self, N=1000):
        groups = self.aa_groups
        positions = pd.Series(self.positions)

        # 重采样
        sample_groups = {}
        group_counts = np.unique([len(group) for group in groups])
        log.debug("开始采样，采样数目为%s", N)
        for count in group_counts:
            # 采样
            one_sample = [positions.sample(n=count, random_state=self.seed + j + count * 1000).tolist() for j in
                          range(N)]
            sample_groups[count] = one_sample
            log.info(f"采样完成 group len =  {count}")
        return sample_groups

    def display_seq_and_aa(self):
        aas = self.non_duplicated_aas
        aas = sorted(aas, key=lambda x: int(x[1:-1]))
        log.info(f"fasta({len(self.fasta)}): {self.fasta}")
        count = [len(group) for group in self.aa_groups_position]
        log.debug("pd.value_counts(count).sort_index() = %s", pd.value_counts(count).sort_index())

        log.info("number of variation: %s", len(self.aa_groups))
        log.info("number of aas: %s", len(self.non_duplicated_aas))
        log.info("number of site: %s", len(self.non_duplicated_aas_positions))


class ProConNetwork:
    def __init__(self,
                 analysis_mutation_groups: AnalysisMutationGroup,
                 data_dir="../data/procon",
                 parse1="../data/procon/type1_parse.csv",
                 parse2="../data/procon/type2_parse.csv",
                 threshold=100,  # 共保守性的阈值
                 ):
        self.analysis_mutation_group = analysis_mutation_groups

        log.info("构造图...")
        # procon 计算的所有的结果
        self.type1 = pd.read_csv(parse1)  # 单点
        self.type2 = pd.read_csv(parse2)  # 成对
        # 归一化保守性分数
        self.type1_mms, self.type1["info_norm"] = self._normalize_info(self.type1["information"].values)
        self.type2_mms, self.type2["info_norm"] = self._normalize_info(self.type2["info"].values)
        # 网络阈值
        if threshold > 1:
            self.threshold = threshold
        else:
            # 按比例划取阈值
            count_type2 = int(threshold * len(self.type2))
            threshold_score = self.type2["info"].sort_values(ascending=False)[count_type2]
            self.threshold = threshold_score
        log.debug("self.threshold = %s", self.threshold)

        # 输出文件路径
        self.data_dir = os.path.join(data_dir, f"threshold_{threshold}")
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)
        log.debug("self.data_dir = %s", self.data_dir)

        # fasta 序列，以及位点
        self.fasta = self.analysis_mutation_group.fasta
        self.positions = self.analysis_mutation_group.positions

        # 构建网络
        nodes = self._get_nodes(self.type1)  # 节点
        links = self._get_links(self.type2[self.type2["info"] >= self.threshold])  # 边
        self.G = self._get_G(links, nodes)  # 构成图
        # 添加相邻位点的最低共保守性
        self.G = self._add_neighbour_links(self.G)

        log.debug(f"不同数据的数目:\n\ttype1 {len(self.type1)}\n\tnode {self.G.number_of_nodes()}\n\t"
                  f"type2 {len(self.type2)}\n\tedge {self.G.number_of_edges()}\n\t"
                  f"neighbour edge {self.G.number_of_edges() - len(links)}")

        # 计算中心性
        log.info("确定中心性...")
        self.degree_c, self.betweenness_c, self.closeness_c, self.edge_betweenness_c = self._get_centralities()

        # page ranke
        log.info("page rank ...")
        self.page_rank = nx.pagerank(self.G)

    @staticmethod
    def _normalize_info(info: np.ndarray):
        """
        归一化数据到 0 和 1 之间
        :return: 归一器 和 当前数据归一结果
        """
        info = info.reshape(-1, 1)
        mms = preprocessing.MinMaxScaler().fit(info)
        return mms, mms.transform(info)

    @staticmethod
    def _aa2position(aa: str):
        if aa[0] in AA and aa[-1] in aa and aa[1:-1].isdigit():
            return aa[1:-1] + aa[0]
        elif aa[:-1].isdigit() and aa[-1] in aa:
            return aa
        else:
            raise RuntimeError(f"无法转化aa={aa}")

    def _get_nodes(self, type1):
        nodes = self.positions
        nodes = pd.DataFrame({"position": nodes})
        nodes = pd.merge(nodes, type1, how="left", left_on="position", right_on="position")
        nodes = nodes.loc[:, ["position", "info_norm"]]
        nodes.columns = ["name", "size"]
        nodes = nodes.fillna(0)
        return nodes

    def _get_links(self, type2, ):
        links = type2.loc[:, ["site1", "site2", "info_norm"]]
        links.columns = ["source", "target", "weight"]
        return links

    def _add_neighbour_links(self, G):
        neighbour_links = []
        for i in range(len(self.positions) - 1):
            node1 = self.positions[i]
            node2 = self.positions[i + 1]
            if not G.has_edge(node1, node2):
                neighbour_links.append([node1, node2, self.threshold])

        neighbour_links = pd.DataFrame(neighbour_links, columns=["source", "target", "weight"])
        neighbour_links["weight"] = self.type2_mms.transform(neighbour_links["weight"].values.reshape(-1, 1), )
        log.debug("neighbour_links = %s", neighbour_links)

        # 添加节点
        for i, row in neighbour_links.iterrows():
            G.add_edge(row["source"], row["target"], weight=row["weight"])
        return G

    def _get_G(self, links, nodes):
        # 绘制关系图
        G = nx.Graph()
        for i, row in nodes.iterrows():
            if row["name"] in self.analysis_mutation_group.non_duplicated_aas_positions:
                G.add_node(row["name"], size=row["size"], is_mutation=True, pst=int(row["name"][:-1]))
            else:
                G.add_node(row["name"], size=row["size"], is_mutation=False, pst=int(row["name"][:-1]))

        for i, row in links.iterrows():
            G.add_edge(row["source"], row["target"], weight=row["weight"])
        return G

    def display(self):
        log.debug("self.type1 = %s", self.type1)
        log.debug("self.type2 = %s", self.type2)
        log.info("nodes = %s", self.nodes)
        log.info("links = %s", self.links)
        log.debug("dc = %s", self.degree_c)
        log.debug("cc = %s", self.closeness_c)
        log.debug("bc = %s", self.betweenness_c)

    @staticmethod
    def count_distribution(d: dict):
        values = list(d.values())
        counts = pd.value_counts(values)
        counts = counts.sort_index()
        counts = counts.to_dict()
        return counts.keys(), counts.values()

    def _get_centralities(self, is_weight=True):
        file_names = ["degree", "betweenness", "closeness", "e_betweenness"]
        # 是否使用带全边
        if is_weight:
            weight = "weight"
            outpath = [os.path.join(self.data_dir, "cache", f"{i}_centrality.json") for i in
                       file_names]
        else:
            weight = None
            outpath = [os.path.join(self.data_dir, "cache", f"{i}_centrality_no_weight.json") for i in
                       file_names]
        for path in outpath:
            if not os.path.exists(os.path.dirname(path)):
                os.mkdir(os.path.dirname(path))
        # 点度中心性 degree
        self.centrality_cache_dir = outpath

        if not os.path.exists(outpath[0]):
            log.debug("无缓存，直接计算")
            log.info("点度中心性 degree")
            dc = degree_centrality(self.G)
            with open(outpath[0], "w") as f:
                f.write(json.dumps(dc))
        else:
            log.debug("存在缓存，直接读取")
            log.info("点度中心性 degree")
            with open(outpath[0], ) as f:
                dc = json.load(f)

        if not os.path.exists(outpath[1]):
            log.debug("无缓存，直接计算")
            # 中介中心性 betweenness
            log.info("中介中心性 betweenness")
            bc = betweenness_centrality(self.G, weight=weight)
            with open(outpath[1], "w") as f:
                f.write(json.dumps(bc))
        else:
            log.debug("存在缓存，直接读取")
            # 中介中心性 betweenness
            log.info("中介中心性 betweenness")
            with open(outpath[1], ) as f:
                bc = json.load(f)

        if not os.path.exists(outpath[2]):
            log.debug("无缓存，直接计算")
            # 接近中心性 closeness
            log.info("接近中心性 closeness")
            cc = closeness_centrality(self.G, distance=weight)
            with open(outpath[2], "w") as f:
                f.write(json.dumps(cc))
        else:
            log.debug("存在缓存，直接读取")
            # 接近中心性 closeness
            log.info("接近中心性 closeness")
            with open(outpath[2], ) as f:
                cc = json.load(f)

        if not os.path.exists(outpath[3]):
            log.debug("无缓存，直接计算")
            # 边 betweenness
            log.info("边 中介中心性 betweenness")
            e_bc = edge_betweenness_centrality(self.G, weight=weight, normalized=False)
            with open(outpath[3], "wb") as f:
                pickle.dump(e_bc, f)
        else:
            log.debug("存在缓存，直接读取")
            # 边 betweenness
            log.info("边 中介中心性 betweenness")
            with open(outpath[3], "rb") as f:
                e_bc = pickle.load(f)

        result = [dc, bc, cc, e_bc]
        # result = [scaler(i) for i in result]
        result = tuple(result)
        return result

    def analysis_aa(self, aa: str):
        aa = self._aa2position(aa)
        res = {
            "degree_centrality": self.degree_c[aa],
            "betweenness_centrality": self.betweenness_c[aa],
            "closeness_centrality": self.betweenness_c[aa]
        }
        return res

    def analysis_group(self, group: list):
        # 单点
        # 箱线图-三种中心性、保守性
        aas = [self._aa2position(i) for i in group]

    def get_degree(self):
        degrees = self.G
        return degrees

    def get_weighted_degree(self):
        """获得加权度"""
        degrees = {}
        # 遍历节点
        for n, nbrs in self.G.adj.items():
            # n 节点，nbrs 邻居节点
            wts = []
            for nbr, eattr in nbrs.items():
                # nbr 邻居节点，eatter 边属性
                wt = eattr["weight"]
                wts.append(wt)
            if wts:
                avg_wt = np.sum(wts)
            else:
                avg_wt = 0.0
            # degrees.append(avg_wt)
            degrees[n] = avg_wt
        return degrees

    def get_avg_weighted_degree(self):
        """获得平均加权度"""
        degrees = {}
        # 遍历节点
        for n, nbrs in self.G.adj.items():
            # n 节点，nbrs 邻居节点
            wts = []
            for nbr, eattr in nbrs.items():
                # nbr 邻居节点，eatter 边属性
                wt = eattr["weight"]
                wts.append(wt)
            if wts:
                avg_wt = np.mean(wts)
            else:
                avg_wt = 0.0
            # degrees.append(avg_wt)
            degrees[n] = avg_wt
        return degrees

    def apply_to_aas(self, aas, aas_sample, func):
        aas_score = func(aas)
        aas_sample_score = [func(a_sample) for a_sample in aas_sample]
        return aas_score, aas_sample_score

    def _plot_degree_distribuition(self, ):
        """
        度分布 degree
        使用边的权重的平均值作为度
        """
        aas = self.analysis_mutation_group.non_duplicated_aas_positions
        aas_sample = self.analysis_mutation_group.non_duplicated_aas_sample

        def calculate_avg_weighted_degree(aas):
            """计算一组位点的平均度"""
            result = []
            for aa in aas:
                # 计算单个位点的加权平均度
                weighted_degrees = []
                for nbr, datadict in self.G.adj[aa].items():
                    weighted_degrees.append(datadict.get("weight", 0))

                if weighted_degrees:
                    avg_weighted_degree = np.mean(weighted_degrees)
                else:
                    avg_weighted_degree = 0
                result.append(avg_weighted_degree)
            return result

        avg_weighted_degrees_result = self.apply_to_aas(aas, aas_sample, calculate_avg_weighted_degree)
        # log.debug("avg_weighted_degrees_result[0] = %s", avg_weighted_degrees_result[0])
        # log.debug("avg_weighted_degrees_result[0][1] = %s", avg_weighted_degrees_result[1][1])
        aas_degrees = pd.Series(avg_weighted_degrees_result[0])
        sample_degrees = pd.Series(np.array(avg_weighted_degrees_result[1]).reshape(-1))
        log.debug("aas_degrees.shape = %s", aas_degrees.shape)
        log.debug("sample_degrees.shape = %s", sample_degrees.shape)
        log.debug("min(aas_degrees) = %s", min(aas_degrees))
        log.debug("max(aas_degrees) = %s", max(aas_degrees))
        log.debug("min(sample_degrees) = %s", min(sample_degrees))
        log.debug("max(sample_degrees) = %s", max(sample_degrees))

        # 寻找最值
        min_value = min(min(aas_degrees), min(sample_degrees))
        max_value = max(max(aas_degrees), max(sample_degrees))
        # 计算坐标分割
        min_value = math.floor(min_value * 10) / 10
        max_value = math.ceil(max_value * 10) / 10 + 0.03
        cut_split = np.arange(min_value, max_value, 0.1).tolist()
        log.debug("cut_split = %s", cut_split)

        # 折线图
        # aas_degrees.sort_values().reset_index(drop=True).plot()
        # plt.show()
        # sample_degrees.sort_values().reset_index(drop=True).plot()
        # plt.show()
        # plt.loglog(aas_degrees.sort_values().values)
        # plt.show()
        # plt.loglog(sample_degrees.sort_values().values)
        # plt.show()

        # 柱状图
        # aas_degrees_cut = pd.cut(aas_degrees, cut_split).value_counts().sort_index() / len(aas_degrees)
        # sample_degrees_cut = pd.cut(sample_degrees, cut_split).value_counts().sort_index() / len(sample_degrees)
        # log.debug("aas_degrees_cut = %s", aas_degrees_cut)
        # log.debug("sample_degrees_cut = %s", sample_degrees_cut)
        # 绘图
        fig: plt.Figure
        axes: List[plt.Axes]
        fig, axes = plt.subplots(1, 2, figsize=(6.4, 4.8), sharex=True, sharey=True, constrained_layout=True)
        # aas_degrees_cut.plot.bar(ax=axes[0])
        sns.histplot(aas_degrees, stat="probability", ax=axes[0], bins=cut_split)
        axes[0].set_xlabel("variant")
        # axes[0].set_xticklabels(aas_degrees_cut.index, rotation=0)

        # sample_degrees_cut.plot.bar(ax=axes[1])
        sns.histplot(sample_degrees, stat="probability", ax=axes[1], bins=cut_split)
        axes[1].set_xlabel("sample")
        # axes[1].set_xticklabels(sample_degrees_cut.index, rotation=0)

        fig.show()
        # fig.tight_layout()
        fig.savefig(os.path.join(self.data_dir, f"度分布.png"), dpi=300)

    def _plot_node_centraility_box(self, ):
        aas = self.analysis_mutation_group.non_duplicated_aas_positions
        aas_sample = self.analysis_mutation_group.non_duplicated_aas_sample
        aas_sample = np.array(aas_sample).reshape(-1).tolist()  # 扁平化
        plot_data = pd.DataFrame(
            {"position": aas + aas_sample, "is_variant": [True] * len(aas) + [False] * len(aas_sample)})
        plot_data["tag"] = plot_data["is_variant"].apply(lambda x: "variant" if x else "sample")

        nodes_size = {node: self.G.nodes[node]["size"] for node in self.G.nodes}
        node_data = pd.DataFrame(
            {"conservation": nodes_size, "degree centrality": self.degree_c, "closeness centrality": self.closeness_c,
             "betweenness centrality": self.betweenness_c})

        # node_data = node_data[]
        plot_data["conservation"] = node_data.loc[plot_data["position"], "conservation"].values
        plot_data["degree centrality"] = node_data.loc[plot_data["position"], "degree centrality"].values
        plot_data["closeness centrality"] = node_data.loc[plot_data["position"], "closeness centrality"].values
        plot_data["betweenness centrality"] = node_data.loc[plot_data["position"], "betweenness centrality"].values
        log.debug("plot_data = %s", plot_data)

        def cal_p_mannwhitneyu(data: pd.Series, bo: pd.Series):
            x = data[bo].to_list()
            y = data[bo.apply(lambda x: not x)].to_list()
            p = mannwhitneyu(x, y).pvalue
            return p

        fig: plt.Figure
        axes: List[plt.Axes]
        fig, axes = plt.subplots(1, 3, figsize=(11, 4.8), constrained_layout=True)
        sns.boxplot(y=plot_data["degree centrality"], x=plot_data["tag"], ax=axes[0], fliersize=1)
        p = cal_p_mannwhitneyu(plot_data["degree centrality"], plot_data["is_variant"])
        # axes[0].set_title(f"degree centrality (p = {p: .3f})")
        axes[0].set_title(f"p = {p: .3f}")
        axes[0].set_xlabel("degree centrality")
        axes[0].set_ylim(axes[0].get_ylim()[0], 0.55)

        sns.boxplot(y=plot_data["closeness centrality"], x=plot_data["tag"], ax=axes[1], fliersize=1)
        p = cal_p_mannwhitneyu(plot_data["closeness centrality"], plot_data["is_variant"])
        # axes[1].set_title(f"closeness centrality (p = {p: .3f})")
        axes[1].set_title(f"p = {p: .3f}")
        axes[1].set_xlabel(f"closeness centrality")

        sns.boxplot(y=plot_data["betweenness centrality"], x=plot_data["tag"], ax=axes[2], fliersize=1)
        p = cal_p_mannwhitneyu(plot_data["betweenness centrality"], plot_data["is_variant"])
        axes[2].set_title(f"p = {p: .3f}")
        axes[2].set_xlabel(f"betweenness centrality")
        axes[2].set_ylim(axes[2].get_ylim()[0], 0.0145)

        [ax.set_ylabel("") for ax in axes]
        # [ax.set_xlabel("") for ax in axes]
        plt.show()
        fig.savefig(os.path.join(self.data_dir, "centraility boxplot.png"), dpi=500)

    def _plot_conservation_box(self, ):
        """计算边的相关参数
        分为三类:
        1. 在同一组的 same mutation group
        2. 在所有可能变异的 mutation
        3. 其他 other
        4. 所有 all
        """
        # 初始化 图片
        fig: plt.Figure
        axes: List[plt.Axes]
        fig, axes = plt.subplots(1, 2, constrained_layout=True, figsize=(7, 4.8))
        # CR, 参考 centrality 函数
        aas = self.analysis_mutation_group.non_duplicated_aas_positions
        aas_sample = self.analysis_mutation_group.non_duplicated_aas_sample
        aas_sample = np.array(aas_sample).reshape(-1).tolist()  # 扁平化
        plot_data = pd.DataFrame(
            {"position": aas + aas_sample, "is_variant": [True] * len(aas) + [False] * len(aas_sample)})
        plot_data["tag"] = plot_data["is_variant"].apply(lambda x: "variant" if x else "sample")

        nodes_size = {node: self.G.nodes[node]["size"] for node in self.G.nodes}
        node_data = pd.DataFrame(
            {"conservation": nodes_size, "degree centrality": self.degree_c, "closeness centrality": self.closeness_c,
             "betweenness centrality": self.betweenness_c})

        plot_data["conservation"] = node_data.loc[plot_data["position"], "conservation"].values

        def cal_p_mannwhitneyu(data: pd.Series, bo: pd.Series):
            x = data[bo].to_list()
            y = data[bo.apply(lambda x: not x)].to_list()
            p = mannwhitneyu(x, y).pvalue
            return p

        sns.boxplot(y=plot_data["conservation"], x=plot_data["tag"], ax=axes[0], fliersize=1)
        p = cal_p_mannwhitneyu(plot_data["conservation"], plot_data["is_variant"])
        axes[0].set_title(f"p = {p: .3f}")
        axes[0].set_xlabel(f"conservation")
        axes[0].set_ylabel("")

        # CCR
        aas = self.analysis_mutation_group.non_duplicated_aas_positions
        groups = self.analysis_mutation_group.aa_groups_position
        # 可以在不同组的边
        edge_in_different_group = list(permutations(aas, 2))
        # 在同一组的边
        edge_in_same_group = []
        for group in groups:
            group = [self._aa2position(i) for i in group]
            edge_in_same_group += list(permutations(group, 2))
        edge_in_same_group = list(set(edge_in_same_group))  # 去重

        def cal_p_mannwhitneyu(data: pd.Series, x: pd.Series, y: pd.Series):
            x = data.loc[x].to_list()
            y = data.loc[y].to_list()
            p = mannwhitneyu(x, y).pvalue
            return p

        # 共保守性
        rows = []
        columns = ["u", "v", "co-conservation", "label"]
        for u, v, weight in self.G.edges.data("weight"):

            # 所有的
            # rows.append([u, v, weight, "all"])

            # 其他三种
            # 先小范围
            if (u, v) in edge_in_same_group:
                rows.append([u, v, weight, "same variant group"])
            # 再大范围和其他
            if (u, v) in edge_in_different_group:
                rows.append([u, v, weight, "variant"])
            else:
                rows.append([u, v, weight, "sample"])
        co_conservation = pd.DataFrame(rows, columns=columns)
        co_conservation = co_conservation[co_conservation["label"] != "same variant group"]
        log.debug("co_conservation.label.value_counts() = %s", co_conservation.label.value_counts())
        sns.boxplot(data=co_conservation, y="co-conservation", x="label", ax=axes[1], fliersize=1,
                    order=["variant", "sample", ],
                    # hue_order=["sampled nodes", "variant nodes"]
                    )
        p_value = cal_p_mannwhitneyu(
            co_conservation["co-conservation"],
            co_conservation["label"] == "variant nodes",
            co_conservation["label"] == "sampled nodes")
        log.debug("p_value = %s", p_value)
        log.debug("p_value = %s", p_value)
        axes[1].set_title(f"p = {p_value:.3f}")
        axes[1].set_xlabel(f"co-conservation")
        axes[1].set_ylabel("")

        [ax.set_ylabel("") for ax in axes]
        # [ax.set_xlabel("") for ax in axes]
        # 绘图
        fig.show()
        fig.savefig(os.path.join(self.data_dir, "edge_boxplot.png"), dpi=500)

    def _plot_procon_distribution(self, ):
        """绘制保守性分数的分布图"""
        log.debug("self.type1.min() = %s", self.type1["information"].min())
        log.debug("self.type1.max() = %s", self.type1["information"].max())
        log.debug("self.type2.min() = %s", self.type2["info"].min())
        log.debug("self.type2.max() = %s", self.type2["info"].max())
        type1_info = self.type1["info_norm"]
        type2_info = self.type2["info_norm"][self.type2["info"] > self.threshold]
        data = pd.DataFrame({"score": pd.concat([type1_info, type2_info]).reset_index(drop=True),
                             "kind": ["CR"] * len(type1_info) + ["CCR"] * len(type2_info)})
        axes: List[plt.Axes]
        fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, constrained_layout=True, figsize=(6.4, 4.8))
        sns.histplot(data=data[data["kind"] == "CR"], x="score", stat="probability", bins=10, ax=axes[0], )
        sns.histplot(data=data[data["kind"] == "CCR"], x="score", stat="probability", bins=10, ax=axes[1], )
        # axes[0].set_xlabel("")
        # axes[1].set_xlabel("")
        axes[0].set_xlabel("conservation")
        axes[1].set_xlabel("co-conservation")
        fig.show()
        fig.savefig(os.path.join(self.data_dir, "procon distribution.png"), dpi=500)

    def _collect_mutation_info(self, save=True):
        """收集变异节点信息"""
        aas = self.analysis_mutation_group.non_duplicated_aas_positions
        weighted_degrees = self.get_weighted_degree()
        avg_weighted_degrees = self.get_avg_weighted_degree()

        data = []
        for aa in aas:
            # 度
            degree = self.G.degree[aa]
            # 加权度
            w_degree = weighted_degrees[aa]
            # 加权平均度
            avg_w_degrees = avg_weighted_degrees[aa]
            # page rank
            page_rank = self.page_rank[aa]
            # conservation
            if aa in self.type1.position.to_list():
                conservation = self.type1["information"][self.type1.position == aa].to_list()[0]
                norm_conservation = self.type1["info_norm"][self.type1.position == aa].to_list()[0]
            else:
                log.debug("%s 不在列表中", aa)
                conservation = 0
                norm_conservation = 0

            # 中心性
            # self.closeness_c[]
            # self.betweenness_c[]
            # self.closeness_c[]

            data.append(
                {"aa": aa, "degree": degree, "weighted degree": w_degree, "average weighted degree": avg_w_degrees,
                 "conservation": conservation, "normalized conservation": norm_conservation,
                 "closeness centrality": self.closeness_c[aa],
                 "betweenness centrality": self.betweenness_c[aa],
                 "degree centrality": self.degree_c[aa],
                 "page rank": page_rank
                 })

        data = pd.DataFrame(data)
        data = data.set_index("aa", drop=False).sort_index()
        if save:
            data.to_csv(os.path.join(self.data_dir, "aas_info.csv"))
        return data

    def calculate_average_shortest_path_length(self, ):
        """平均最短路径长度"""
        # data -> aa : [变异，非变异]
        aas = self.analysis_mutation_group.non_duplicated_aas_positions
        aas_sample = self.analysis_mutation_group.non_duplicated_aas_sample
        aas_scores = []
        sample_scores = []
        for aa in aas:
            # 构造数据
            path_len = shortest_path_length(self.G, source=aa)
            path_len = pd.Series(path_len)
            # 在变异中
            aas_scores.append(path_len[aas].mean())
            # 在采样中
            sample_scores.append([path_len[a_sample].mean() for a_sample in aas_sample])

        plot_data = pd.DataFrame(sample_scores, index=aas, )
        plot_data = plot_data.stack().reset_index()
        plot_data.columns = ["aa", "group", "length"]
        # 保存详细数据
        detail_data = pd.DataFrame(
            {"aa": aas, "aas_avg_shortest_length": aas_scores, "sample_avg_shortest_length": sample_scores,
             "position": [int(i[:-1]) for i in aas]}
        ).sort_values(by="position", ).to_csv(os.path.join(self.data_dir, "averrage shortest length.csv"))

        fig: plt.Figure
        # 根据平均最短路径，绘制箱线图并与单值进行比较
        # axes: List[plt.Axes]
        ax: plt.Axes
        fig, ax = plt.subplots(1, 1, figsize=(14, 4.8))
        # 查看并保存图片
        order_index = np.argsort(aas_scores)
        sns.boxplot(data=plot_data, x="aa", y="length", ax=ax, order=np.array(aas)[order_index], fliersize=1)
        ax.scatter(x=range(len(aas)), y=np.array(aas_scores)[order_index])
        # ax.set_ylabel("average shortest path length")
        ax.set_ylabel("")
        ax.set_xlabel("")
        [txt.set_rotation(90) for txt in ax.get_xticklabels()]

        fig.tight_layout()
        fig.show()
        fig.savefig(os.path.join(self.data_dir, "average shortest length.png"), dpi=300)

        # 热力图  => 效果很不好，考虑删除
        # 判断变异位点的平均最短路径是否高于上四分位线，如果是则为1否则为0
        # 热力图中，以毒株为x，以变异为y
        # 如果变异在毒株中且值为1则绿，否则无色
        # 判断是否大于分位数
        is_bigger = {}
        for i in range(len(aas)):
            aa = aas[i]
            score = aas_scores[i]
            s_scores = sample_scores[i]
            percentile_75 = np.percentile(s_scores, 75)
            is_bigger[aa] = score >= percentile_75
        log.debug("is_bigger = %s", is_bigger)
        # 拿到热力图数据
        heatmap_data = []
        aa_groups = self.analysis_mutation_group.aa_groups_position
        aa_groups_names = self.analysis_mutation_group.aa_groups_info["name"]
        for group in aa_groups:
            row = []
            for aa, flag in is_bigger.items():
                if aa in group:
                    if flag:
                        row.append(1)
                    else:
                        row.append(0)
                else:
                    row.append(0)
            heatmap_data.append(row)
        heatmap_data = pd.DataFrame(heatmap_data, index=aa_groups_names, columns=aas)
        log.debug("heatmap_data = %s", heatmap_data)
        # 绘图
        fig = plt.figure(figsize=(10, 15))
        ax = fig.subplots()
        sns.heatmap(heatmap_data, ax=ax)
        fig.tight_layout()
        fig.show()

        # 柱状图
        # 毒株中高于四分位线位点的变异占据的比例
        is_bigger_position = set([k for k, v in is_bigger.items() if v])
        is_bigger_ratio = []
        for group in aa_groups:
            is_bigger_in_group = set(group) & is_bigger_position
            is_bigger_ratio.append(
                len(is_bigger_in_group) / len(group)
            )
        log.debug("is_bigger_ratio = %s", is_bigger_ratio)
        barplot_data = pd.DataFrame({"y": is_bigger_ratio, })
        barplot_data = pd.concat([barplot_data, self.analysis_mutation_group.aa_groups_info], axis=1)
        log.debug("barplot_data = %s", barplot_data)
        # 绘图
        fig = plt.figure(figsize=(15, 10))
        ax = fig.subplots()
        sns.barplot(data=barplot_data, x="name", y="y", hue="category")
        [txt.set_rotation(90) for txt in ax.get_xticklabels()]
        fig.tight_layout()
        fig.show()
        fig.savefig(os.path.join(self.data_dir, "平均最短路径长度 较大值占据毒株比例.png"), dpi=300)

    def analysisG(self, ):
        """绘制相关图表"""
        # self._plot_origin_distribution()  # 绘制所有节点的保守性的分布情况
        # self._plot_mutations_relationship()  # 绘制变异位点的关系图: 节点-变异位点，节点大小-出现的次数，边-是否存在共保守性
        # self._collect_mutation_info()  # 收集变异位点的消息，生成表格
        # self._plot_2D()  # 二维坐标图
        #
        # self._plot_procon_distribution()  # 分数分布图
        # self._plot_degree_distribuition()  # 度分布
        # self._plot_node_centraility_box()  # 箱线图：中心性
        # self._plot_conservation_box()  # 保守性
        # self.calculate_average_shortest_path_length()
        #
        # # # 以组为单位的图
        self._group_plot_with_node()

    def output_for_gephi(self):
        # 边
        rows = []
        for source, target in self.G.edges:
            is_neighbour = abs(int(target[:-1]) - int(source[:-1])) == 1
            is_mutation = (source in self.analysis_mutation_group.non_duplicated_aas_positions) or (
                    target in self.analysis_mutation_group.non_duplicated_aas_positions)
            tag = "normal"
            if is_neighbour:
                tag = "neighbour"
            elif is_mutation:
                tag = "mutation"

            w = self.G.edges[source, target]['weight']
            if tag == "neighbour" and w < 0.9:
                w = 0.9

            rows.append([source, target, is_neighbour, is_mutation, tag, w])
        rows = pd.DataFrame(rows, columns=["source", "target", "is_neighbour", "is_mutation", "tag", "weight"], )
        rows.to_csv(os.path.join(self.data_dir, "network_edge_info.csv"), index=None)

        # 节点
        nodes = []
        for n in self.G.nodes:
            if n in self.analysis_mutation_group.non_duplicated_aas_positions:
                node = [n, n, True, 20]
            else:
                node = [n, "", False, 10]
            nodes.append(node)
        nodes = pd.DataFrame(nodes, columns=["Id","Label", "is_mutation", "size"])
        nodes.to_csv(os.path.join(self.data_dir, "network_node_info.csv"), index=None)

    def output_for_DynaMut2(self):
        for i, (group, name) in enumerate(zip(self.analysis_mutation_group.aa_groups,
                                              self.analysis_mutation_group.aa_groups_info["name"]
                                              )):
            name = name.replace("/", "or")
            with open(os.path.join(self.data_dir, f"dynamut2 input v4 ({i} {name}).txt"), "w") as f:
                for a1, a2 in combinations(group, 2):
                    if "X" in a1 or "X" in a2:
                        continue
                    if name == "Omicron":
                        if self.G.has_edge(self._aa2position(a1), self._aa2position(a2)):
                            f.write(f"A {a1};A {a2}\n")
                    else:
                        f.write(f"A {a1};A {a2}\n")

    def random_sample_analysis(self, aas: list, groups, N=1000):
        """使用随机采样的形式，分析实验组和对照组的区别
        :param N: 采样次数
        """
        aas = [self._aa2position(aa) for aa in aas if aa]
        aas = list(set(aas))
        groups = [[self._aa2position(aa) for aa in group] for group in groups]
        log.debug("aas = %s", aas)

        # 重采样
        group_and_sample_groups = []
        positions = pd.Series(self.positions)
        log.info("开始采样，采样数目为%s", N)
        for group in groups:
            # 采样
            sample_groups = [positions.sample(n=len(group), random_state=i).tolist() for i in range(N)]
            group_and_sample_groups.append([group, sample_groups])
            log.info("采样完成")

        def apply_to_group(data, func):
            scores = []
            for group, sample_groups in data:
                s1 = func(group)
                s2s = [func(g) for g in sample_groups]
                scores.append([s1, s2s])
            return scores

        # 度
        # weighted_degrees = self.get_weighted_degree()
        # avg_weighted_degrees = self.get_avg_weighted_degree()

        def calculate_degree(group: List[str]):
            """计算平均度"""
            data = []
            for aa in group:
                degree = self.G.degree[aa]
                data.append(degree)

            # 返回
            return np.mean(data)

        degree_scores = apply_to_group(group_and_sample_groups, calculate_degree)
        # log.debug("degree_scores = %s", degree_scores)
        # log.debug("degree_scores[0][1] = %s", degree_scores[0][1])
        # for i in degree_scores:
        #     print(type(i[1]))

        _plot_scores = []
        for i in degree_scores:
            _plot_scores += i[1]
        _plot_aas = []
        for i in range(len(group_and_sample_groups)):
            _plot_aas += [i, ] * N
        log.debug("len(_plot_aas) = %s", len(_plot_aas))
        log.debug("len(_plot_scores) = %s", len(_plot_scores))
        log.debug("len(degree_scores) = %s", len(degree_scores))
        log.debug("group_and_sample_groups = %s", len(group_and_sample_groups))
        assert len(_plot_aas) == len(_plot_scores)
        # 绘图的数据
        plot_data = pd.DataFrame(
            np.array([_plot_scores, _plot_aas]).T,
            columns=["degree", "position"]
        )
        log.debug("plot_data = %s", plot_data)
        ax: plt.Axes
        fig, ax = plt.subplots()
        sns.boxplot(data=plot_data, x="position", y="degree", ax=ax)
        _plot_normal_degree = [i[0] for i in degree_scores]
        ax.scatter(x=list(range(len(group_and_sample_groups))), y=_plot_normal_degree)
        [txt.set_rotation(90) for txt in ax.get_xticklabels()]
        fig.tight_layout()
        fig.show()
        fig.savefig(os.path.join(self.data_dir, "resample degree.png"), dpi=300)

    def _plot_origin_distribution(self):
        type2 = self.type2["info"]
        type2_count = pd.cut(type2, np.arange(max(type2))).value_counts().sort_index()
        type2_count.index = np.arange(max(type2))[:len(type2_count)]

        # plt.plot(x=type2_count.index, y=type2_count.values)
        ax: plt.Axes
        fig, ax = plt.subplots(figsize=(6.4, 4.8))
        type2_count.plot(ax=ax)
        # ax.set_xtick(np.arange(max(type2)))
        ax.set_title("")
        ax.set_ylabel("density")
        fig.tight_layout()
        plt.show()
        fig.savefig(os.path.join(self.data_dir, "共保守性分布情况.png"), dpi=300)

    def _plot_mutations_relationship(self):
        groups = self.analysis_mutation_group.aa_groups
        aas = [self.analysis_mutation_group.aa2position(aa) for group in groups for aa in group]
        # # 节点
        # unique_aas = np.unique(aas)
        # 节点和权重: 权重统计出现的次数
        aas_count = pd.value_counts(aas)
        # 边: 判断是否具有共保守性
        edges = []
        for n1, n2 in combinations(aas_count.index, 2):
            if self.G.has_edge(n1, n2):
                edges.append([n1, n2, self.G.edges[n1, n2]["weight"]])
        # # 导出数据，让其他软件绘图
        # nodes = pd.DataFrame({"ID": aas_count.index, "weight": aas_count.values})
        # nodes.index.name = "rank"
        # links = pd.DataFrame(links, columns=["Source", "Target", "weight"])
        # links.index.name = "rank"
        #
        # nodes.to_csv(os.path.join(self.data_dir, "mutation relationship.gepi.node.csv"))
        # links.to_csv(os.path.join(self.data_dir, "mutation relationship.gepi.link.csv"))

        # # 使用networkx
        # log.debug("aas_count = %s", aas_count.shape)
        # log.debug("links = %s", len(links))
        # G = nx.Graph()
        # for node, weight in aas_count.items():
        #     G.add_node(node, weight=weight)
        # for n1, n2, weight in links:
        #     G.add_edge(n1, n2, weight=weight)
        # fig = plt.figure(figsize=(20, 20), )
        # nx.draw_circular(G, )
        # nx.draw_networkx_labels(G, nx.drawing.circular_layout(G, scale=1.05), font_size=3, font_color="g", )
        # fig.tight_layout()
        # fig.show()
        # fig.savefig(os.path.join(self.data_dir, "mutation count relationships.png"), dpi=300)

        # 使用 echart 绘图
        nodes = pd.DataFrame({"name": aas_count.index, "symbolSize": aas_count.values})
        nodes = nodes.to_dict("records")
        links = pd.DataFrame(edges, columns=["source", "target", "weight"])
        links.to_csv(os.path.join(self.data_dir, "test_links.csv"))
        links = links.to_dict("records")
        count_node_in_links = defaultdict(int)
        for link in links:
            source = link["source"]
            target = link["target"]
            count_node_in_links[source] += 1
            count_node_in_links[target] += 1
        log.debug("pd.Series(count_node_in_links) = %s", pd.Series(count_node_in_links).sort_values())

        c = (
            Graph(init_opts=opts.InitOpts(width="100%", height="1000px"))
                .add("", nodes, links, repulsion=8000, layout="circular", )
                .set_global_opts(
                title_opts=opts.TitleOpts(title="count"),
                toolbox_opts=opts.ToolboxOpts(
                    feature=opts.ToolBoxFeatureOpts(
                        save_as_image=opts.ToolBoxFeatureSaveAsImageOpts(pixel_ratio=3, background_color="white"))), )
                .render(os.path.join(self.data_dir, "mutation relationship.html"))
        )

    def _group_plot_with_node(self):
        groups = self.analysis_mutation_group.aa_groups_position
        groups_names = self.analysis_mutation_group.aa_groups_info["name"]
        groups_colors = self.analysis_mutation_group.aa_groups_info["color"]
        group_count_sample = self.analysis_mutation_group.group_count_sample

        # # TODO 采样 10000 次
        # group_count_sample = self.analysis_mutation_group.resample_groups(10000)

        def calculate_group_and_sample_score(grp, grp_sample, func, fig_name, kind="distribution", excel_writer=None):
            grp_scores = [func(i) for i in grp]
            grp_mean_score = [np.mean(i) for i in grp_scores]
            grp_sample_scores = {}
            for count, sample_group in grp_sample.items():
                sample_scores = [func(group) for group in sample_group]
                sample_mean_score = np.mean(sample_scores, axis=1)
                sample_mean_score = sorted(sample_mean_score)  # 排序
                grp_sample_scores[count] = sample_mean_score

            # log.debug("grp_sample_scores = %s", grp_sample_scores)
            grp_info = pd.DataFrame({"name": groups_names, "score": grp_mean_score, "color": groups_colors})
            grp_info["length"] = [len(g) for g in grp]
            grp_info["index"] = np.arange(len(grp_info)) + 1
            log.debug("np.unique(grp_info.length) = %s", np.unique(grp_info.length))

            # TODO: 合并 27 BA.2 和 28 BA.2+ L452X
            # 修改 BA.2+ L452X 数量为 27
            index_count_27_28 = (grp_info["length"] == 27) | (grp_info["length"] == 28)
            grp_info.loc[index_count_27_28, "length"] = "27 or 28"
            grp_sample_scores["27 or 28"] = grp_sample_scores[27]
            grp_sample_scores.pop(27)
            grp_sample_scores.pop(28)
            Ns = grp_sample_scores.keys()
            Ns = sorted(Ns, key=lambda x: int(re.match("\d+", str(x)).group(0)))  # 匹配第一个数字

            # 只需要 7组
            assert grp_info["length"].value_counts().shape[0] == 7

            # 绘制图表
            if kind == "distribution":
                """绘制采样分数的分布图，并将毒株标注在图中"""
                fig: plt.Figure = plt.figure(figsize=(20, 20))
                axes: List[plt.Axes] = fig.subplots(3, 3, )
                colors = sns.color_palette(n_colors=len(grp_sample_scores))
                axes = [j for i in axes for j in i]
                ax_all_in_one = axes[7]

                statistic_table = []
                for i, N in enumerate(Ns):
                    # 绘制分布图
                    color = colors[i]
                    ax: plt.Axes = axes[i]
                    sample_mean_score = grp_sample_scores[N]
                    # sns.distplot(sample_mean_score, ax=ax, color=color)
                    # sns.distplot(sample_mean_score, ax=ax_all_in_one, color=color)

                    sns.histplot(sample_mean_score, ax=ax, color=color, kde=True, stat="probability")
                    sns.histplot(sample_mean_score, ax=ax_all_in_one, color=color, kde=True, stat="probability")

                    # 统计数据: 分位数和平均数
                    statistic_data = {
                        "a quarter of a quintile": np.percentile(sample_mean_score, 25),
                        "one half quantile": np.percentile(sample_mean_score, 50),
                        "three quarters of a quintile": np.percentile(sample_mean_score, 75),
                        "mean": np.mean(sample_mean_score)
                    }

                    # 处理当前N的每个毒株
                    current_n_grp_info = grp_info[grp_info["length"] == N]
                    current_n_grp_info = current_n_grp_info.sort_values("score", ascending=False)
                    for index, row in current_n_grp_info.iterrows():
                        _x = [row["score"]] * len(sample_mean_score)
                        _y = range(1, len(sample_mean_score) + 1)
                        ax.axvline(row["score"], ls="-", label=row["name"], color=row["color"])
                        ax_all_in_one.axvline(row["score"], ls="-", label=row["name"], color=row["color"])

                        # 收集每组一些统计数据
                        t, pvalue = ttest_1samp(sample_mean_score, row["score"])
                        if pvalue < 0.05:
                            if t < 0:
                                is_remarkable = "Yes. Left."
                            else:
                                is_remarkable = "Yes. Right."
                        else:
                            is_remarkable = "No"

                        statistic_data["score"] = row["score"]
                        statistic_data["t"] = t
                        statistic_data["p"] = pvalue
                        statistic_data["result"] = is_remarkable
                        statistic_data["name"] = row["name"]
                        statistic_data["index"] = row["index"]

                        statistic_table.append(statistic_data.copy())

                    ax.set_title(f"N = {N}", y=-0.1)
                    ax.legend()
                ax_all_in_one.set_title(f"all", y=-0.1)
                # ax_all_in_one.legend()
                # ax_all_in_one.get_legend().remove()

                # 箱线图
                _s1 = grp_mean_score  # 每一组的分数
                _s2 = np.array(list(grp_sample_scores.values())).reshape(-1).tolist()  # 采样的分数
                _plot_data = pd.DataFrame(
                    {"score": _s1 + _s2, "label": ["variant"] * len(_s1) + ["sample"] * len(_s2)},
                )
                sns.boxplot(data=_plot_data, x="label", y="score", ax=axes[-1], fliersize=1)
                p_value = mannwhitneyu(_s1, _s2).pvalue
                axes[-1].set_title(f"p = {p_value:.3f}", y=-0.1)
                axes[-1].set_xlabel("")

                # 给global加上箱线图
                global_axes = self.group_global_axes[self.group_global_ax_count]
                self.group_global_ax_count += 1
                sns.boxplot(data=_plot_data, x="label", y="score", ax=global_axes, fliersize=1)
                # global_axes.set_xlabel(f"{fig_name} (p={p_value:.3f})", y=-0.1)
                global_axes.set_title(f"p={p_value:.3f}", )
                global_axes.set_xlabel(f"{fig_name}", )
                # global valid
                if p_value <= 0.05:
                    global_axes = self.group_global_valid_axes[self.group_global_valid_ax_count]
                    self.group_global_valid_ax_count += 1
                    sns.boxplot(data=_plot_data, x="label", y="score", ax=global_axes, fliersize=1)
                    # self.group_global_valid_fig.show()
                    global_axes.set_xlabel(f"{fig_name} (p={p_value:.3f})", y=-0.1)

                # 输出结果
                fig.suptitle(fig_name, )
                fig.tight_layout()
                fig.show()
                fig.savefig(os.path.join(self.data_dir, f"group distribution {fig_name}.png"), dpi=300)

                # 存储统计信息
                if excel_writer:
                    statistic_table = pd.DataFrame(statistic_table)
                    statistic_table = statistic_table.sort_values("index")
                    statistic_table.to_excel(excel_writer, sheet_name=fig_name)

            elif kind == "score_sorted":
                """ 直接绘制排序后的分数（废弃，只做备份）"""
                fig: plt.Figure = plt.figure(figsize=(20, 20))
                axes: List[plt.Axes] = fig.subplots(3, 3, )
                axes = [j for i in axes for j in i]
                ax_all_in_one = axes[7]  # 重叠子图
                # _s2 = []
                for i, N in enumerate(grp_sample_scores.keys()):
                    ax: plt.Axes = axes[i]
                    sample_mean_score = grp_sample_scores[N]
                    # _s2 += grp_info[grp_info["length"] == N].shape[0] * np.array(sample_mean_score).reshape(-1).tolist()
                    ax.plot(range(1, len(sample_mean_score) + 1), sample_mean_score)
                    ax_all_in_one.plot(range(1, len(sample_mean_score) + 1), sample_mean_score)
                    for index, row in grp_info[grp_info["length"] == N].sort_values("score",
                                                                                    ascending=False).iterrows():
                        ax.plot(range(1, len(sample_mean_score) + 1), [row["score"]] * len(sample_mean_score),
                                label=row["name"])
                        ax_all_in_one.plot(range(1, len(sample_mean_score) + 1),
                                           [row["score"]] * len(sample_mean_score),
                                           label=row["name"])
                        # ax.loglog(range(1, len(sample_mean_score) + 1), [row["score"]] * len(sample_mean_score))
                        # ax.text(x=0, y=row["score"], s=row["name"])

                    ax.set_title(f"N = {N}", y=-0.1)
                    ax.legend()
                ax_all_in_one.set_title(f"all", y=-0.1)
                # 箱线图
                _s1 = grp_mean_score  # 每一组的分数
                _s2 = np.array(list(grp_sample_scores.values())).reshape(-1).tolist()  # 采样的分数
                _plot_data = pd.DataFrame(
                    {"score": _s1 + _s2, "label": ["mutation"] * len(_s1) + ["sample"] * len(_s2)},
                )
                sns.boxplot(data=_plot_data, x="label", y="score", ax=axes[-1])

                p_value = mannwhitneyu(_s1, _s2).pvalue
                axes[-1].set_title(f"p = {p_value:.3f}", y=-0.1)
                axes[-1].set_xlabel("")

                fig.suptitle(fig_name, )
                fig.tight_layout()
                fig.show()
                fig.savefig(os.path.join(self.data_dir, f"group {fig_name}.png"), dpi=300)

        def calculate_degree_centrality(grp):
            dc = scaler(self.degree_c)
            return [dc[aa] for aa in grp]

        def calculate_betweenness_centrality(grp):
            bc = scaler(self.betweenness_c)
            return [bc[aa] for aa in grp]

        def calculate_closeness_centrality(grp):
            cc = scaler(self.closeness_c)
            return [cc[aa] for aa in grp]

        def calculate_avg_weighted_degree(grp):
            """计算一组位点的平均度"""
            result = []
            for aa in grp:
                # 计算单个位点的加权平均度
                weighted_degrees = []
                for nbr, datadict in self.G.adj[aa].items():
                    weighted_degrees.append(datadict.get("weight", 0))

                if weighted_degrees:
                    avg_weighted_degree = np.mean(weighted_degrees)
                else:
                    avg_weighted_degree = 0
                result.append(avg_weighted_degree)
            return result

        def calculate_page_rank(grp):
            pr = scaler(self.page_rank)
            return [pr[aa] for aa in grp]

        def calculate_conservation(grp):
            return [self.G.nodes[aa]["size"] for aa in grp]

        def calculate_weighted_shortest_path(grp):
            res = []
            if not hasattr(self, "weighted_shortest_path_length"):
                # log.info("没有 self.weighted_shortest_path_length, 初始化")
                # log.debug(" = %s", dict(nx.shortest_path_length(self.G, weight="weight")))
                # log.debug(" = %s", dict(nx.shortest_path_length(self.G, )))
                # self.weighted_shortest_path_length = dict(nx.shortest_path_length(self.G, weight="weight"))
                # with open(os.path.join(self.data_dir, "weighted shortest path length.json"), "w") as f:
                #     log.info("保存至文件")
                #     f.write(json.dumps(self.weighted_shortest_path_length))

                with open(os.path.join(self.data_dir, "weighted shortest path length.json"), ) as f:
                    log.info("加载文件")
                    self.weighted_shortest_path_length = json.loads(f.read())

                log.debug("初始化完成")
            # 归一器
            if not hasattr(self, "scaler_for_weighted_shortest_path_length"):
                all_values = []
                for v1 in self.weighted_shortest_path_length.values():
                    for v2 in v1.values():
                        all_values.append(v2)
                all_values = np.array(all_values).reshape((-1, 1))
                self.scaler_for_weighted_shortest_path_length = preprocessing.MinMaxScaler().fit(all_values)

            wspl = self.weighted_shortest_path_length
            for n1, n2 in combinations(grp, 2):
                res.append(
                    self.scaler_for_weighted_shortest_path_length.transform([[wspl[n1][n2]]])[0][0])
            return res

        def calculate_edge_betweenness_centrality(grp):
            res = []
            # if not hasattr(self, "edge_betweenness_centrality"):
            #     try:
            #         # 保存至 json 文件
            #         if not os.path.exists(os.path.join(self.data_dir, "edge betweenness centrality.json")):
            #             log.info("没有 self.edge_betweenness_centrality, 初始化")
            #             self.edge_betweenness_centrality = dict(nx.edge_betweenness_centrality(self.G, weight="weight"))
            #             # 重新建立索引
            #             info_map = defaultdict(dict)
            #             for (n1, n2), value in self.edge_betweenness_centrality.items():
            #                 info_map[n1][n2] = value
            #                 info_map[n2][n1] = value
            #
            #             with open(os.path.join(self.data_dir, "edge betweenness centrality.json"), "w") as f:
            #                 log.info("保存至文件")
            #                 f.write(json.dumps(info_map))
            #
            #         with open(os.path.join(self.data_dir, "edge betweenness centrality.json"), ) as f:
            #             log.info("加载文件")
            #             self.edge_betweenness_centrality = json.loads(f.read())
            #     except:
            #         log.error("保存或加载文件失败")
            #     log.debug("初始化完成")
            #
            # ebc = self.edge_betweenness_centrality
            # # ebc_values = []
            # # for k1, v1 in ebc.items():
            # #     for k2, v2 in v1.items():
            # #         ebc_values.append(v2)
            # # scaler = preprocessing.MinMaxScaler().fit(np.array(ebc_values))
            #
            # for n1, n2 in combinations(grp, 2):
            #     if n1 in ebc and n2 in ebc[n1]:
            #         res.append(ebc[n1][n2])
            #     elif n2 in ebc and n1 in ebc[n2]:
            #         res.append(ebc[n2][n1])
            #     else:
            #         res.append(0)
            if not hasattr(self, "edge_betweenness_centrality_scaler"):
                self.edge_betweenness_centrality_scaler = scaler(self.edge_betweenness_c)

            ebc = self.edge_betweenness_centrality_scaler
            for n1, n2 in combinations(grp, 2):
                if (n1, n2) in self.edge_betweenness_c:
                    res.append(ebc[(n1, n2)])
                elif (n2, n1) in self.edge_betweenness_c:
                    res.append(ebc[(n2, n1)])
                else:
                    res.append(0)
            return res

        def calculate_co_conservation_rate():
            """使用闭包将索引放置在函数内"""
            # 找到共保守性 pairwise，并建立索引
            # type2 = self.type2[self.type2["info"] >= self.threshold]
            type2 = self.type2
            idx = defaultdict(list)
            for i, row in type2.iterrows():
                idx[row.site1].append(row.site2)
                idx[row.site2].append(row.site1)

            def _cal(grp):
                # 较强共共保守性占的比例
                rns = len(grp)  # 氨基酸个数
                rnp = comb(rns, 2)  # pairwise 个数
                rpc = 0  # 共保守性 pairwise 个数
                for p1, p2 in combinations(grp, 2):
                    if p2 in idx[p1]:
                        rpc += 1
                res = [rpc / rnp, ] * len(grp)
                return res

            return _cal

        # 单独将箱线图拿出
        self.group_global_fig: plt.Figure = plt.figure(figsize=(14, 9.6))
        self.group_global_axes = [j for i in self.group_global_fig.subplots(2, 4) for j in i]
        self.group_global_ax_count = 0
        # 只绘制 p < 0.05 的图
        self.group_global_valid_fig: plt.Figure = plt.figure(figsize=(16, 8))
        self.group_global_valid_axes = [j for i in self.group_global_valid_fig.subplots(3, 3) for j in i]
        self.group_global_valid_ax_count = 0

        # 统计信息表
        excel_writer = pd.ExcelWriter(os.path.join(self.data_dir, "group distribution statistic information.xlsx"))

        # 关于节点
        calculate_group_and_sample_score(groups, group_count_sample, calculate_degree_centrality, "degree centrality",
                                         excel_writer=excel_writer)
        calculate_group_and_sample_score(groups, group_count_sample, calculate_betweenness_centrality,
                                         "betweenness centrality", excel_writer=excel_writer)
        calculate_group_and_sample_score(groups, group_count_sample, calculate_closeness_centrality,
                                         "closeness centrality", excel_writer=excel_writer)
        calculate_group_and_sample_score(groups, group_count_sample, calculate_avg_weighted_degree,
                                         "average weighted degree", excel_writer=excel_writer)
        calculate_group_and_sample_score(groups, group_count_sample, calculate_page_rank, "page rank",
                                         excel_writer=excel_writer)
        calculate_group_and_sample_score(groups, group_count_sample, calculate_conservation, "conservation",
                                         excel_writer=excel_writer)

        # 关于边
        calculate_group_and_sample_score(groups, group_count_sample, calculate_weighted_shortest_path,
                                         "shortest weighted path length", excel_writer=excel_writer)
        calculate_group_and_sample_score(groups, group_count_sample, calculate_edge_betweenness_centrality,
                                         "edge betweenness centrality", excel_writer=excel_writer)
        # # 关于 高保守性
        # calculate_group_and_sample_score(groups, group_count_sample, calculate_co_conservation_rate(),
        #                                  "rate of co-pairwise", excel_writer=excel_writer)

        # 调整 global 图
        # title 为 xlabel; 删除 xlabel ylabel
        for axes in [self.group_global_axes, self.group_global_valid_axes]:
            axes: List[plt.Axes]
            # [ax.set_title(ax.get_xlabel()) for ax in axes]
            # [ax.set_xlabel("") for ax in axes]
            [ax.set_ylabel("") for ax in axes]
        self.group_global_fig.tight_layout()
        self.group_global_fig.savefig(os.path.join(self.data_dir, "group distribution global.png"), dpi=300)
        [i.set_visible(False) for i in self.group_global_valid_axes[self.group_global_valid_ax_count:]]  # 删除多余子图
        self.group_global_valid_fig.tight_layout()
        self.group_global_valid_fig.savefig(os.path.join(self.data_dir, "group distribution global valid.png"), dpi=300)

        excel_writer.close()

    def _plot_2D(self, font_size="x-large", txt_rotation=0, x_rotation=90):
        analysis = self.analysis_mutation_group.analysis
        type1 = []  # 变体中单位点保守性
        type2 = []  # 变体中共保守性
        name2index = {}  # 变体名 -> 索引
        co_conservation_rate = []  # 较强高保守性的比例
        for i, (_, row) in enumerate(analysis.items()):
            # 所有出现的变异  type1
            t1 = pd.DataFrame(row["type1"])
            t1["y"] = i
            # 寻找名称
            if type(row["WHO label"]) == str:
                name = "{}({})".format(row["Lineage + additional mutations"], row["WHO label"], )
            else:
                name = "{}".format(row["Lineage + additional mutations"], )
            t1["name"] = name
            name2index[name] = i

            # type2
            t2 = row["type2"]
            t2 = pd.DataFrame(t2)
            t2["y1"] = t2["y2"] = i  # 两点的纵坐标

            type1.append(t1)
            type2.append(t2)

            # 较强共共保守性占的比例
            rns = len(t1)
            rnp = comb(rns, 2)
            rpc = len(t2)
            rr = rpc / rnp
            co_conservation_rate.append(
                {
                    "name": name,
                    "N(substitution)": rns,
                    "N(pairwise)": rnp,
                    "N(pairwise with co-conservation)": rpc,
                    "rate": rr
                }
            )
        # 保存比例
        co_conservation_rate = pd.DataFrame(co_conservation_rate)
        co_conservation_rate.to_csv(
            os.path.join(self.data_dir, "rate of pairwise with co-conservation.csv")
        )

        # 原始数据
        type1 = pd.concat(type1)
        type2 = pd.concat(type2)

        """
        图片可以分为四个部分
            * 横坐标
            * 纵坐标
            * 图中的点
            * 图中的线
        """

        # 图中点和线的横坐标
        # 找到所有的位点，并排序确定索引
        type1["idx"] = type1["aa"].str[1:-1].astype(int)
        # position -> 横坐标
        pst_2_x = type1["idx"].drop_duplicates().sort_values().to_list()
        pst_2_x = {j: i for i, j in enumerate(pst_2_x)}
        # 确定横坐标
        type1["x"] = type1["idx"].map(pst_2_x)
        type2["x1"] = type2["site1"].str[:-1].astype(int).map(pst_2_x)
        type2["x2"] = type2["site2"].str[:-1].astype(int).map(pst_2_x)

        # 图中的线的中点
        # 曲线最高点: 找到第三点的坐标
        gap = 1
        type2["x3"] = (type2["x1"] + type2["x2"]) / 2
        log.debug("type2 = %s", type2)
        type2["y3"] = (type2["y1"] + gap / 4) + (type2["x2"] - type2["x1"]).abs() / (len(pst_2_x) - 2)

        # # 边
        # type2_info = type2.loc[:, ["site1", "site2", "info", "rate"]].drop_duplicates(["site1", "site2"])
        # log.debug("type2_info = %s", type2_info)

        # 绘图
        def two_degree_bc(x, y, dots_num=100):  # bezier curve
            """二阶贝塞尔曲线"""
            x1, x2, x3 = x
            y1, y2, y3 = y
            xt = []  # 目标点的x坐标
            yt = []  # 目标点的y坐标
            x_dots12 = np.linspace(x1, x2, dots_num)  # 线段AB的x坐标
            y_dots12 = np.linspace(y1, y2, dots_num)  # 线段AB的y坐标
            x_dots23 = np.linspace(x2, x3, dots_num)  # 线段BC的x坐标
            y_dots23 = np.linspace(y2, y3, dots_num)  # 线段BC的y坐标
            for i in range(dots_num):  # 获得目标点的轨迹
                x = x_dots12[i] + (x_dots23[i] - x_dots12[i]) * i / (dots_num - 1)
                y = y_dots12[i] + (y_dots23[i] - y_dots12[i]) * i / (dots_num - 1)
                xt.append(x)
                yt.append(y)
            xt = np.array(xt)
            yt = np.array(yt)
            return xt, yt

        # 初始化图表
        # flg_size = [max(len(pst_2_x), len(analysis)), ] * 2
        # fig_size = np.array(flg_size) / 2.6
        fig_size = (28, 14)
        print("fig_size:", fig_size)
        flg, ax = plt.subplots(figsize=fig_size)

        # 添加文字：变异
        for i, row in type1.sort_values(["y", ]).iterrows():
            x, y, txt = row["x"], row["y"], row["aa"]
            ax.text(x, y, txt[-1], ha="center", va="center", size=font_size, rotation=txt_rotation)

        # 绘制曲线
        for i, row in type2.iterrows():
            x1, x2, x3, y1, y2, y3 = row[["x1", "x2", "x3", "y1", "y2", "y3"]]
            x = [x1, x3, x2]
            y = [y1, y3, y2]
            x_smooth, y_smooth = two_degree_bc(x, y)
            y_smooth += 0.15
            alpha = row["rate"] / 100
            ax.plot(x_smooth, y_smooth, "C1", alpha=alpha * 1.5)
            # ax.plot(x_smooth, y_smooth, "C1",)

        flg.show()
        # 添加坐标标签
        xtick_names = type1["position"].drop_duplicates()
        arg_sort = xtick_names.str[:-1].astype(int).argsort()
        xtick_names = xtick_names.iloc[arg_sort]
        ytick_names = type1["name"].drop_duplicates()
        arg_sort = ytick_names.apply(lambda x: name2index[x]).astype(int).argsort()
        ytick_names = ytick_names.iloc[arg_sort]

        ax.set_xticks(range(len(xtick_names)))
        ax.set_yticks(range(len(ytick_names)))
        ax.set_xticklabels(xtick_names, size=font_size)
        ax.set_yticklabels(ytick_names, size=font_size)
        # ax.invert_yaxis()  # 反转y轴
        # ax.set_ymargin(0.03)
        # ax.set_xmargin(-0.1)
        ax.margins(0.02)
        [txt.set_rotation(x_rotation) for txt in ax.get_xticklabels()]

        # 绘制表格
        flg.tight_layout()
        flg.show()
        flg.savefig(os.path.join(self.data_dir, "2D relationship.png"), dpi=300)

    def output_for_topnettree(self):
        # 找出不同的变异
        log.info(self.analysis_mutation_group.non_duplicated_aas)

    def generate_ebc(self):
        if not hasattr(self, "edge_betweenness_centrality"):
            try:
                # 保存至 json 文件
                if not os.path.exists(os.path.join(self.data_dir, "edge betweenness centrality.json")):
                    log.info("没有 self.edge_betweenness_centrality, 初始化")
                    self.edge_betweenness_centrality = dict(nx.edge_betweenness_centrality(self.G, weight="weight"))
                    # 重新建立索引
                    info_map = defaultdict(dict)
                    for (n1, n2), value in self.edge_betweenness_centrality.items():
                        info_map[n1][n2] = value
                        info_map[n2][n1] = value

                    with open(os.path.join(self.data_dir, "edge betweenness centrality.json"), "w") as f:
                        log.info("保存至文件")
                        f.write(json.dumps(info_map))

                with open(os.path.join(self.data_dir, "edge betweenness centrality.json"), ) as f:
                    log.info("加载文件")
                    self.edge_betweenness_centrality = json.loads(f.read())
            except:
                log.error("保存或加载文件失败")
            log.debug("初始化完成")

if __name__ == '__main__':
    start_time = time.time()
    # 保守性网络
    # 需要关注的变异
    mutation_groups = AnalysisMutationGroup()
    mutation_groups.display_seq_and_aa()
    pcn = ProConNetwork(mutation_groups, threshold=100)
    log.debug("len(pcn.type2) = %s", len(pcn.type2))
    # pcn.analysisG()
    # pcn.generate_ebc()
    # print(pd.value_counts([len(i) for i in pcn.analysis_mutation_group.aa_groups]))  # 统计变体中变异数量
    # pcn.output_for_gephi()
    # pcn.output_for_DynaMut2()
    # pcn.output_for_topnettree()
    end_time = time.time()
    log.info(f"程序运行时间: {end_time - start_time}")
    print(mutation_groups.non_duplicated_aas)

    # thresholds = [50, 100, 150, 200, 250, 300]
    # for t in thresholds:
    #     if t != 100:
    #         continue
    #     pcn = ProConNetwork(mutation_groups, threshold=t)
    #
    #     pcn.analysisG()
    #     # pcn.random_sample_analysis(aas, groups.get_aa_groups())
    #
    #     end_time = time.time()
    #     log.info(f"程序运行时间: {end_time - start_time}")
