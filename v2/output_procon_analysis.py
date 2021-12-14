import json
import math

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
from scipy.stats import mannwhitneyu
from itertools import combinations, permutations
from collections import defaultdict
from brokenaxes import brokenaxes
from matplotlib.gridspec import GridSpec
from sklearn import preprocessing
# 日志
import logging
import logconfig
import pickle
from typing import List

logconfig.setup_logging()
log = logging.getLogger("cov2")

AA = ['-', 'C', 'F', 'I', 'L', 'M', 'V', 'W', 'Y', 'A', 'T', 'D', 'E', 'G', 'P', 'N', 'Q', 'S', 'H', 'R', 'K']

sns.set()


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
            categroies.append(row["Unnamed: 9"])
            if type(row["WHO label"]) == str:
                # name = "{} ({})".format(row["Lineage + additional mutations"], row["WHO label"])
                name = "{}".format(row["WHO label"])
            else:
                name = "{}".format(row["Lineage + additional mutations"])
            names.append(name)

        info = pd.DataFrame({"name": names, "category": categroies})
        # 为类别设置颜色
        color_map = dict(zip(np.unique(categroies), ["#EDD2F3", "#FFFCDC", "#84DFFF", "#516BEB"]))
        info["color"] = info["category"].map(color_map)
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
        # log.info(f"fasta:\n"
        #          f"{self.fasta}\n" + "aas:\n" + "\n".join(aas)
        #          )
        log.debug("len(self.positions) = %s", len(self.positions))


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
            G.add_node(row["name"], size=row["size"])

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
            e_bc = edge_betweenness_centrality(self.G, weight=weight)
            with open(outpath[3], "wb") as f:
                pickle.dump(e_bc, f)
        else:
            log.debug("存在缓存，直接读取")
            # 边 betweenness
            log.info("边 中介中心性 betweenness")
            with open(outpath[3], "rb") as f:
                e_bc = pickle.load(f)

        return dc, bc, cc, e_bc

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
        # 进行归一化
        min_value = math.floor(min_value * 10) / 10
        max_value = math.ceil(max_value * 10) / 10 + 0.03
        cut_split = np.arange(min_value, max_value, 0.1).tolist()
        log.debug("cut_split = %s", cut_split)

        # 折线图
        aas_degrees.sort_values().reset_index(drop=True).plot()
        plt.show()
        sample_degrees.sort_values().reset_index(drop=True).plot()
        plt.show()
        plt.loglog(aas_degrees.sort_values().values)
        plt.show()
        plt.loglog(sample_degrees.sort_values().values)
        plt.show()

        # 柱状图
        aas_degrees_cut = pd.cut(aas_degrees, cut_split).value_counts().sort_index() / len(aas_degrees)
        sample_degrees_cut = pd.cut(sample_degrees, cut_split).value_counts().sort_index() / len(sample_degrees)
        log.debug("aas_degrees_cut = %s", aas_degrees_cut)
        log.debug("sample_degrees_cut = %s", sample_degrees_cut)
        # 绘图
        fig: plt.Figure
        axes: List[plt.Axes]
        fig, axes = plt.subplots(1, 2, figsize=(15, 7.5))
        aas_degrees_cut.plot.bar(ax=axes[0])
        axes[0].set_title("degree distribution of mutations")
        axes[0].set_xticklabels(aas_degrees_cut.index, rotation=0)

        sample_degrees_cut.plot.bar(ax=axes[1])
        axes[1].set_title("degree distribution")
        axes[1].set_xticklabels(sample_degrees_cut.index, rotation=0)

        fig.show()
        fig.savefig(os.path.join(self.data_dir, f"度分布.png"), dpi=300)
        # TODO: 绘制箱线图

    def _plot_node_box(self, ):
        aas = self.analysis_mutation_group.non_duplicated_aas_positions
        aas_sample = self.analysis_mutation_group.non_duplicated_aas_sample
        aas_sample = np.array(aas_sample).reshape(-1).tolist()  # 扁平化
        plot_data = pd.DataFrame(
            {"position": aas + aas_sample, "is_mutation": [True] * len(aas) + [False] * len(aas_sample)})

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
        fig, axes = plt.subplots(4, 1, figsize=(10, 15))
        sns.boxplot(x=plot_data["conservation"], y=plot_data["is_mutation"], orient="h", ax=axes[0], )
        p = cal_p_mannwhitneyu(plot_data["conservation"], plot_data["is_mutation"])
        axes[0].set_xlabel(f"conservtion (p = {p: .3f})")
        sns.boxplot(x=plot_data["degree centrality"], y=plot_data["is_mutation"], orient="h", ax=axes[1], )
        p = cal_p_mannwhitneyu(plot_data["degree centrality"], plot_data["is_mutation"])
        axes[1].set_xlabel(f"degree centrality (p = {p: .3f})")
        sns.boxplot(x=plot_data["closeness centrality"], y=plot_data["is_mutation"], orient="h", ax=axes[2])
        p = cal_p_mannwhitneyu(plot_data["closeness centrality"], plot_data["is_mutation"])
        axes[2].set_xlabel(f"closeness centrality (p = {p: .3f})")
        sns.boxplot(x=plot_data["betweenness centrality"], y=plot_data["is_mutation"], orient="h", ax=axes[3])
        p = cal_p_mannwhitneyu(plot_data["betweenness centrality"], plot_data["is_mutation"])
        axes[3].set_xlabel(f"betweenness centrality (p = {p: .3f})")
        plt.show()
        fig.savefig(os.path.join(self.data_dir, "boxplot.png"), dpi=500)

    def _plot_edge_box(self, ):
        """计算边的相关参数
        分为三类:
        1. 在同一组的 same mutation group
        2. 在所有可能变异的 mutation
        3. 其他 other
        4. 所有 all

        """
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

        # 初始化 图片
        fig: plt.Figure
        fig, ax = plt.subplots(1, 1, )

        # 共保守性
        rows = []
        columns = ["u", "v", "co-conservation", "label"]
        for u, v, weight in self.G.edges.data("weight"):

            # 所有的
            # rows.append([u, v, weight, "all"])

            # 其他三种
            # 先小范围
            if (u, v) in edge_in_same_group:
                rows.append([u, v, weight, "same mutation group"])
            # 再大范围和其他
            if (u, v) in edge_in_different_group:
                rows.append([u, v, weight, "mutation"])
            else:
                rows.append([u, v, weight, "no mutation"])
        co_conservation = pd.DataFrame(rows, columns=columns)
        co_conservation = co_conservation[co_conservation["label"] != "same mutation group"]
        log.debug("co_conservation.label.value_counts() = %s", co_conservation.label.value_counts())
        sns.boxplot(data=co_conservation, x="co-conservation", y="label", orient="h", ax=ax)
        p_value = cal_p_mannwhitneyu(
            co_conservation["co-conservation"],
            co_conservation["label"] == "mutation",
            co_conservation["label"] == "no mutation")
        log.debug("p_value = %s", p_value)
        # p_value = cal_p_mannwhitneyu(
        #     co_conservation["co-conservation"],
        #     co_conservation["label"] == "same mutation group",
        #     co_conservation["label"] == "no mutation")
        log.debug("p_value = %s", p_value)
        ax.set_xlabel(f"co-conservation (p = {p_value:.3f})")
        ax.set_ylabel("")
        # 绘图
        fig.tight_layout()
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

        cut_list = np.arange(0, 1.1, 0.1).tolist()
        log.debug("cut_list = %s", cut_list)
        type1_info_count = pd.cut(type1_info, cut_list, ).value_counts().sort_index() / len(type1_info)
        type2_info_count = pd.cut(type2_info, cut_list, ).value_counts().sort_index() / len(type2_info)
        plot_data = pd.DataFrame([
            type1_info_count.to_list() + type2_info_count.to_list(),
            ["conservation"] * len(type1_info_count) + ["co-conservation"] * len(type2_info_count),
            type1_info_count.index.to_list() + type2_info_count.index.to_list(),
            # ])
        ], index=["proportion", "type", "score"]).T
        fig = sns.catplot(
            kind="bar",
            x="type",
            y="proportion",
            hue="score",
            data=plot_data,
            height=10,
        )
        plt.show()
        fig.savefig(os.path.join(self.data_dir, "procon distribution.png"), dpi=500)

    def _collect_mutation_info(self, ):
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
                 })

        data = pd.DataFrame(data)
        data = data.set_index("aa", drop=False).sort_index()
        data.to_csv(os.path.join(self.data_dir, "aas_info.csv"))

    def calculate_average_shortest_path_length(self, ):
        """平均最短路径长度"""
        # data -> aa : [变异，非变异]
        data = {}
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
        fig: plt.Figure

        # 根据平均最短路径，绘制箱线图并与单值进行比较
        # axes: List[plt.Axes]
        ax: plt.Axes
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        # 查看并保存图片
        order_index = np.argsort(aas_scores)
        sns.boxplot(data=plot_data, x="aa", y="length", ax=ax, order=np.array(aas)[order_index])
        ax.scatter(x=range(len(aas)), y=np.array(aas_scores)[order_index])
        ax.set_ylabel("average shortest path length")
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
        # self._plot_procon_distribution()  # 分数分布图
        # self._plot_degree_distribuition()  # 度分布
        # self._plot_node_box()  # 箱线图：中心性 + 保守性
        # self._plot_edge_box()  # 共保守性  TODO: 使用采样的方式
        # self.calculate_average_shortest_path_length()

        # 以组为单位的图
        self._group_plot_centtrality()

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
        fig, ax = plt.subplots(figsize=(10, 5))
        type2_count.plot(ax=ax)
        # ax.set_xtick(np.arange(max(type2)))
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

    def _group_plot_centtrality(self):
        groups = self.analysis_mutation_group.aa_groups_position
        groups_names = self.analysis_mutation_group.aa_groups_info["name"]
        group_count_sample = self.analysis_mutation_group.group_count_sample

        def calculate_group_and_sample_score(grp, grp_sample, func, fig_name, kind="distribution"):
            grp_scores = [func(i) for i in grp]
            grp_mean_score = [np.mean(i) for i in grp_scores]
            grp_sample_scores = {}
            for count, sample_group in grp_sample.items():
                sample_scores = [func(group) for group in sample_group]
                sample_mean_score = np.mean(sample_scores, axis=1)
                sample_mean_score = sorted(sample_mean_score)  # 排序
                grp_sample_scores[count] = sample_mean_score

            # log.debug("grp_sample_scores = %s", grp_sample_scores)
            grp_info = pd.DataFrame({"name": groups_names, "score": grp_mean_score})
            grp_info["length"] = [len(g) for g in grp]
            log.debug("np.unique(grp_info.length) = %s", np.unique(grp_info.length))
            # 绘制图表
            if kind == "distribution":
                """绘制采样分数的分布图，并将毒株标注在图中"""
                fig: plt.Figure = plt.figure(figsize=(20, 20))
                axes: List[plt.Axes] = fig.subplots(3, 3, )
                axes = [j for i in axes for j in i]
                ax_all_in_one = axes[7]
                for i, N in enumerate(grp_sample_scores.keys()):
                    # 绘制分布图
                    ax: plt.Axes = axes[i]
                    sample_mean_score = grp_sample_scores[N]
                    sns.distplot(sample_mean_score, ax=ax)
                    sns.distplot(sample_mean_score, ax=ax_all_in_one)
                fig.suptitle(fig_name, )
                fig.tight_layout()
                fig.show()
                fig.savefig(os.path.join(self.data_dir, f"group distribution {fig_name}.png"), dpi=300)
            elif kind == "score_sorted":
                """ 直接绘制排序后的分数（废弃，只做备份）"""
                fig: plt.Figure = plt.figure(figsize=(20, 20))
                axes: List[plt.Axes] = fig.subplots(3, 3, )
                axes = [j for i in axes for j in i]
                ax_all_in_one = axes[7]  # 重叠子图
                for i, N in enumerate(grp_sample_scores.keys()):
                    ax: plt.Axes = axes[i]
                    sample_mean_score = grp_sample_scores[N]
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
            return [self.degree_c[aa] for aa in grp]

        def calculate_betweenness_centrality(grp):
            return [self.betweenness_c[aa] for aa in grp]

        def calculate_closeness_centrality(grp):
            return [self.closeness_c[aa] for aa in grp]

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
            return [self.page_rank[aa] for aa in grp]
        def calculate_conservation(grp):
            return [self.G.nodes[aa]["size"] for aa in grp]

        calculate_group_and_sample_score(groups, group_count_sample, calculate_degree_centrality, "degree centrality")
        calculate_group_and_sample_score(groups, group_count_sample, calculate_betweenness_centrality,
                                         "betweenness centrality")
        calculate_group_and_sample_score(groups, group_count_sample, calculate_closeness_centrality,
                                         "closeness centrality")
        calculate_group_and_sample_score(groups, group_count_sample, calculate_avg_weighted_degree,
                                         "average weighted degree")
        calculate_group_and_sample_score(groups, group_count_sample, calculate_page_rank, "page rank")
        calculate_group_and_sample_score(groups, group_count_sample, calculate_conservation, "conservation")


if __name__ == '__main__':
    start_time = time.time()
    # 保守性网络
    # 需要关注的变异
    mutation_groups = AnalysisMutationGroup()
    # mutation_groups.display_seq_and_aa()
    pcn = ProConNetwork(mutation_groups, threshold=100)
    pcn.analysisG()
    end_time = time.time()
    log.info(f"程序运行时间: {end_time - start_time}")

    # thresholds = [50, 100, 150, 200, 250, 300]
    # for t in thresholds:
    #     pcn = ProConNetwork(mutation_groups, threshold=t)
    #
    #     pcn.analysisG()
    #     # pcn.random_sample_analysis(aas, groups.get_aa_groups())
    #
    #     end_time = time.time()
    #     log.info(f"程序运行时间: {end_time - start_time}")
